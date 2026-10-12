from __future__ import annotations

import argparse
import os
import sys

import torch

BIASDY_DIR = os.path.abspath(os.path.dirname(__file__))
DEBIAS_ROOT = os.path.abspath(os.path.join(BIASDY_DIR, ".."))
if BIASDY_DIR not in sys.path:
    sys.path.insert(0, BIASDY_DIR)
if DEBIAS_ROOT not in sys.path:
    sys.path.insert(0, DEBIAS_ROOT)

from dataset import EUROCDataset
from learning import Test
from network_for_BiadDy import ba_func_choose, bw_func_choose
from plato.evaluation import print_metrics_table, summarize_comparison_metrics, write_metrics_csv
from plato.loss_smoke_test import main as loss_smoke_main
from plato.nll_smoke_test import main as nll_smoke_main
from plato.supervision import load_vo_for_sequence
from plato.supervision_smoke_test import main as supervision_smoke_main
from plato.training import (
    PlatoTrainConfig,
    load_ba_func,
    load_bw_func,
    train_mse_acc,
    train_mse_gyro,
    train_mse_joint,
    train_nll,
    train_nll_alternating,
)


DEFAULT_EUROC_TRAIN_VAL_SEQS = [
    "MH_01_easy",
    "MH_03_medium",
    "MH_05_difficult",
    "V1_02_medium",
    "V2_01_easy",
    "V2_03_difficult",
]

DEFAULT_EUROC_TEST_SEQS = [
    "MH_02_easy",
    "MH_04_difficult",
    "V2_02_medium",
    "V1_03_difficult",
    "V1_01_easy",
]


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.debug_smoke:
        loss_smoke_main()
        nll_smoke_main()
        supervision_smoke_main()
        return

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.plato_supervision == "vo_pose" and args.plato_sup_stride is None and args.plato_sup_rate_hz is None:
        args.plato_sup_stride = 40
    if args.loss_window is None:
        args.loss_window = args.plato_window_steps

    dataset_parameters = euroc_dataset_parameters(args)
    dataset_train = EUROCDataset(**dataset_parameters, mode="train", recompute=args.recompute_data)
    dataset_val = EUROCDataset(**dataset_parameters, mode="val")
    dataset_test = EUROCDataset(**dataset_parameters, mode="test")

    config = make_train_config(args)
    vo_train = load_vo_trajectories(dataset_train, args)
    vo_val = load_vo_trajectories(dataset_val, args)

    gyro_checkpoint = args.resume_gyro
    acc_checkpoint = args.resume_acc

    if args.plato_stage in ("gyro", "mse", "both"):
        gyro_checkpoint = train_mse_gyro(
            dataset_train,
            dataset_val,
            config,
            lr=args.lr_Gyro,
            weight_decay=args.weight_decay_Gyro,
            epoch_count=args.epoch_mse,
            gyro_checkpoint=gyro_checkpoint,
            vo_train=vo_train,
            vo_val=vo_val,
        )
        if args.plato_stage in ("mse", "both"):
            acc_checkpoint = train_mse_acc(
                dataset_train,
                dataset_val,
                config,
                gyro_checkpoint=gyro_checkpoint,
                lr=args.lr_Acc,
                weight_decay=args.weight_decay_Acc,
                epoch_count=args.epoch_mse,
                acc_checkpoint=acc_checkpoint,
                vo_train=vo_train,
                vo_val=vo_val,
            )

    if args.plato_stage == "acc":
        if gyro_checkpoint is None:
            raise ValueError("--resume_gyro is required for --plato_stage acc")
        acc_checkpoint = train_mse_acc(
            dataset_train,
            dataset_val,
            config,
            gyro_checkpoint=gyro_checkpoint,
            lr=args.lr_Acc,
            weight_decay=args.weight_decay_Acc,
            epoch_count=args.epoch_mse,
            acc_checkpoint=acc_checkpoint,
            vo_train=vo_train,
            vo_val=vo_val,
        )

    run_joint = args.plato_stage == "joint" or (
        args.plato_stage in ("mse", "both") and args.epoch_joint_mse > 0
    )
    if run_joint:
        if gyro_checkpoint is None:
            raise ValueError("--resume_gyro is required for joint MSE")
        if args.epoch_joint_mse <= 0:
            raise ValueError("--epoch_joint_mse must be positive for --plato_stage joint")
        joint_checkpoint = train_mse_joint(
            dataset_train,
            dataset_val,
            config,
            gyro_checkpoint=gyro_checkpoint,
            acc_checkpoint=acc_checkpoint,
            lr=args.lr_joint_mse,
            weight_decay=args.weight_decay_joint_mse,
            epoch_count=args.epoch_joint_mse,
            vo_train=vo_train,
            vo_val=vo_val,
        )
        gyro_checkpoint = joint_checkpoint
        acc_checkpoint = joint_checkpoint

    if args.plato_stage == "nll" and (gyro_checkpoint is None or acc_checkpoint is None):
        raise ValueError("--resume_gyro and --resume_acc are required for --plato_stage nll")

    nll_checkpoint = None
    if args.plato_stage in ("nll", "both"):
        if args.nll_alternating:
            if args.nll_freeze_gyro or args.nll_freeze_acc:
                raise ValueError("Alternating NLL cannot use one-block freeze flags")
            nll_checkpoint = train_nll_alternating(
                dataset_train, dataset_val, config,
                gyro_checkpoint=gyro_checkpoint, acc_checkpoint=acc_checkpoint,
                bias_lr=args.lr_NLL, covariance_lr=args.lr_covariance,
                bias_weight_decay=args.weight_decay_NLL, covariance_weight_decay=0.0,
                epoch_count=args.epoch_nll, block_epochs=args.nll_block_epochs,
                start_with=args.nll_start_with, vo_train=vo_train, vo_val=vo_val,
            )
        else:
            nll_checkpoint = train_nll(
                dataset_train,
                dataset_val,
                config,
                gyro_checkpoint=gyro_checkpoint,
                acc_checkpoint=acc_checkpoint,
                lr=args.lr_NLL,
                weight_decay=args.weight_decay_NLL,
                epoch_count=args.epoch_nll,
                nll_freeze_gyro=args.nll_freeze_gyro,
                nll_freeze_acc=args.nll_freeze_acc,
                vo_train=vo_train,
                vo_val=vo_val,
            )

    if args.run_test:
        test_gyro = nll_checkpoint if nll_checkpoint is not None else gyro_checkpoint
        test_acc = nll_checkpoint if nll_checkpoint is not None else acc_checkpoint
        if test_gyro is None or test_acc is None:
            raise ValueError("--resume_gyro and --resume_acc are required for --run_test")
        run_test(dataset_test, args, test_gyro, test_acc)

    if args.run_compare:
        rows = summarize_comparison_metrics(dataset_test, args.outputdir)
        csv_path = os.path.join(args.outputdir, "compare", "metrics_summary.csv")
        write_metrics_csv(rows, csv_path)
        print_metrics_table(rows)
        print(f"Wrote metrics summary to {csv_path}")


def build_parser():
    parser = argparse.ArgumentParser("PLATO EuRoC training for Debias_IMU")
    parser.add_argument("--integral_method", choices=["euler"], default="euler")
    parser.add_argument(
        "--mean_discretization",
        choices=["paper_zoh"],
        default="paper_zoh",
        help=(
            "Structure-preserving ZOH state propagation with Euler bias dynamics."
        ),
    )
    parser.add_argument("--outputdir", type=str, default="results/Euroc_PLATO")
    parser.add_argument("--network_type_Gyro", type=str, default="bw_func_net")
    parser.add_argument("--network_type_Acc", type=str, default="ba_func_net")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug_smoke", action="store_true")
    parser.add_argument("--recompute_data", action="store_true")

    parser.add_argument("--data_dir", type=str, default="data/EUROC")
    parser.add_argument("--data_cache_dir", type=str, default="data/EUROC")
    parser.add_argument("--train_seqs", nargs="+", default=DEFAULT_EUROC_TRAIN_VAL_SEQS)
    parser.add_argument("--val_seqs", nargs="+", default=DEFAULT_EUROC_TRAIN_VAL_SEQS)
    parser.add_argument("--test_seqs", nargs="+", default=DEFAULT_EUROC_TEST_SEQS)
    parser.add_argument("--loss_window", type=int, default=None)
    parser.add_argument("--plato_supervision", choices=["gt_se23", "vo_pose"], default="gt_se23")
    parser.add_argument("--plato_stage", choices=["gyro", "mse", "acc", "joint", "nll", "both", "eval"], default="mse")
    parser.add_argument("--plato_window_steps", type=int, default=64)
    parser.add_argument("--plato_sup_stride", type=int, default=None)
    parser.add_argument("--plato_sup_rate_hz", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=1000)

    parser.add_argument("--epoch_mse", type=int, default=1801)
    parser.add_argument("--epoch_joint_mse", type=int, default=0)
    parser.add_argument("--epoch_nll", type=int, default=300)
    parser.add_argument("--lr_Gyro", type=float, default=5e-3)
    parser.add_argument("--lr_Acc", type=float, default=5e-3)
    parser.add_argument("--lr_joint_mse", type=float, default=1e-4)
    parser.add_argument("--lr_NLL", type=float, default=1e-4)
    parser.add_argument("--nll_alternating", action="store_true")
    parser.add_argument("--lr_covariance", type=float, default=5e-5)
    parser.add_argument("--nll_block_epochs", type=int, default=1)
    parser.add_argument("--nll_start_with", choices=["covariance", "bias"], default="covariance")
    parser.add_argument("--ba_rate_anneal_start_fraction", type=float, default=0.6)
    parser.add_argument("--ba_rate_final_limit", type=float, default=5e-3)
    parser.add_argument("--weight_decay_Gyro", type=float, default=1e-6)
    parser.add_argument("--weight_decay_Acc", type=float, default=1e-6)
    parser.add_argument("--weight_decay_joint_mse", type=float, default=1e-6)
    parser.add_argument("--weight_decay_NLL", type=float, default=1e-6)
    parser.add_argument("--val_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--train_sequence_strategy", choices=["random", "cycle"], default="random")
    parser.add_argument("--mse_rot_weight", type=float, default=1.0)
    parser.add_argument("--mse_vel_weight", type=float, default=1.0)
    parser.add_argument("--mse_pos_weight", type=float, default=1.0)
    parser.add_argument("--mse_acc_rot_weight", type=float, default=0.0)
    parser.add_argument("--mse_acc_vel_weight", type=float, default=1.0)
    parser.add_argument("--mse_acc_pos_weight", type=float, default=1.0)
    parser.add_argument("--mse_residual_mode", choices=["log", "right", "global", "hybrid"], default="log")
    parser.add_argument(
        "--mse_gyro_objective",
        choices=["se23", "legacy_so3"],
        default="se23",
        help=(
            "Gyro-stage rotation loss normalization. legacy_so3 exactly matches "
            "the coordinatewise rotation MSE; se23 uses 0.5*mean(||Log(.)||^2)."
        ),
    )
    parser.add_argument(
        "--gradient_mode",
        choices=["autograd", "double_adjoint"],
        default="autograd",
        help="Bias-dynamics gradient mode for gyro/acc/joint MSE and NLL training.",
    )
    parser.add_argument(
        "--covariance_gradient_mode",
        choices=["forward_sensitivity", "autograd"],
        default="forward_sensitivity",
        help="Differentiate learned covariance by explicit forward sensitivity or reverse AD.",
    )
    parser.add_argument("--analytic_grad_scale", type=float, default=1.0)
    parser.add_argument("--analytic_grad_clip_norm", type=float, default=0.0)
    parser.add_argument("--analytic_residual_jacobian", choices=["bch", "identity", "left", "right"], default="bch")
    parser.add_argument("--gyro_multi_window_steps", nargs="+", type=int, default=None)
    parser.add_argument("--gyro_multi_window_weights", nargs="+", type=float, default=None)

    parser.add_argument("--resume_gyro", type=str, default=None)
    parser.add_argument(
        "--resume_acc",
        type=str,
        default=None,
        help=(
            "Optional BA checkpoint. For --plato_stage joint, omitting it "
            "initializes a new BA network while loading BG from --resume_gyro."
        ),
    )
    parser.add_argument("--vo_dir", type=str, default=None)
    parser.add_argument("--vo_template", type=str, default="{seq}/00_estimate.txt")
    parser.add_argument("--vo_align", choices=["first_gt", "none"], default="first_gt")
    parser.add_argument("--vo_residual_mode", choices=["absolute", "relative"], default="absolute")

    parser.add_argument("--meas_sigma_rot_gt", type=float, default=1e-3)
    parser.add_argument("--meas_sigma_vel_gt", type=float, default=1e-2)
    parser.add_argument("--meas_sigma_pos_gt", type=float, default=1e-2)
    parser.add_argument("--meas_sigma_rot_vo", type=float, default=1e-2)
    parser.add_argument("--meas_sigma_pos_vo", type=float, default=5e-2)
    parser.add_argument("--init_sigma_a", type=float, default=0.1)
    parser.add_argument("--init_sigma_g", type=float, default=0.1)
    parser.add_argument("--plato_noise_isotropic", type=str2bool, default=True)
    parser.add_argument("--nll_min_sigma_a", type=float, default=1e-4)
    parser.add_argument("--nll_max_sigma_a", type=float, default=1.0)
    parser.add_argument("--nll_min_sigma_g", type=float, default=1e-5)
    parser.add_argument("--nll_max_sigma_g", type=float, default=0.1)
    parser.add_argument("--cov_jitter", type=float, default=0.0)
    parser.add_argument("--nll_gt_sup_stride", type=int, default=4)
    parser.add_argument("--prior_std", type=float, default=1e-4)
    parser.add_argument("--nll_gt_mode", choices=["woodbury", "sparse"], default="sparse")
    parser.add_argument(
        "--nll_gt_residual_mode",
        choices=["right", "log", "hybrid_global"],
        default="right",
    )
    parser.add_argument(
        "--nll_gt_include_logdet_h",
        type=str2bool,
        default=True,
        help=(
            "Include Eq. (9) sum_i log|det H_i| in sparse right-residual GT "
            "NLL. Set false only for the controlled ablation."
        ),
    )
    parser.add_argument("--nll_vo_mode", choices=["woodbury", "sparse"], default="woodbury")
    parser.add_argument("--nll_mse_anchor_weight", type=float, default=0.0)
    parser.add_argument("--nll_freeze_gyro", action="store_true")
    parser.add_argument("--nll_freeze_acc", action="store_true")

    parser.add_argument("--run_test", action="store_true")
    parser.add_argument("--run_compare", action="store_true")
    parser.add_argument("--test_device", type=str, default="cpu")
    parser.add_argument("--test_recompute", action="store_true")
    parser.add_argument("--test_bias_init_mode", choices=["zero", "first", "mean"], default="first")
    parser.add_argument("--test_bias_init_seconds", type=float, default=5.0)
    return parser


def euroc_dataset_parameters(args):
    return {
        "dataset_name": "EUROC",
        "data_dir": args.data_dir,
        "data_cache_dir": args.data_cache_dir,
        "train_seqs": args.train_seqs,
        "val_seqs": args.val_seqs,
        "test_seqs": args.test_seqs,
        "dt": 0.005,
        "percent_for_val": 0.2,
        "loss_window": args.loss_window,
        "batch_size": args.batch_size,
        "sg_window_bw": 1,
        "sg_order_bw": 0,
        "sg_window_ba": 1,
        "sg_order_ba": 0,
    }


def make_train_config(args):
    return PlatoTrainConfig(
        supervision=args.plato_supervision,
        window_steps=args.plato_window_steps,
        sup_stride=args.plato_sup_stride,
        sup_rate_hz=args.plato_sup_rate_hz,
        integral_method=args.integral_method,
        mean_discretization=args.mean_discretization,
        batch_size=args.batch_size,
        outputdir=args.outputdir,
        network_type_Gyro=args.network_type_Gyro,
        network_type_Acc=args.network_type_Acc,
        meas_sigma_rot_gt=args.meas_sigma_rot_gt,
        meas_sigma_vel_gt=args.meas_sigma_vel_gt,
        meas_sigma_pos_gt=args.meas_sigma_pos_gt,
        meas_sigma_rot_vo=args.meas_sigma_rot_vo,
        meas_sigma_pos_vo=args.meas_sigma_pos_vo,
        init_sigma_a=args.init_sigma_a,
        init_sigma_g=args.init_sigma_g,
        plato_noise_isotropic=args.plato_noise_isotropic,
        nll_min_sigma_a=args.nll_min_sigma_a,
        nll_max_sigma_a=args.nll_max_sigma_a,
        nll_min_sigma_g=args.nll_min_sigma_g,
        nll_max_sigma_g=args.nll_max_sigma_g,
        cov_jitter=args.cov_jitter,
        nll_gt_sup_stride=args.nll_gt_sup_stride,
        prior_std=args.prior_std,
        val_freq=args.val_freq,
        save_freq=args.save_freq,
        device=args.device,
        train_sequence_strategy=args.train_sequence_strategy,
        mse_rot_weight=args.mse_rot_weight,
        mse_vel_weight=args.mse_vel_weight,
        mse_pos_weight=args.mse_pos_weight,
        mse_acc_rot_weight=args.mse_acc_rot_weight,
        mse_acc_vel_weight=args.mse_acc_vel_weight,
        mse_acc_pos_weight=args.mse_acc_pos_weight,
        mse_residual_mode=args.mse_residual_mode,
        mse_gyro_objective=args.mse_gyro_objective,
        ba_rate_anneal_start_fraction=args.ba_rate_anneal_start_fraction,
        ba_rate_final_limit=args.ba_rate_final_limit,
        gradient_mode=args.gradient_mode,
        covariance_gradient_mode=args.covariance_gradient_mode,
        analytic_grad_scale=args.analytic_grad_scale,
        analytic_grad_clip_norm=args.analytic_grad_clip_norm,
        analytic_residual_jacobian=args.analytic_residual_jacobian,
        gyro_multi_window_steps=tuple(args.gyro_multi_window_steps or ()),
        gyro_multi_window_weights=tuple(args.gyro_multi_window_weights or ()),
        nll_gt_mode=args.nll_gt_mode,
        nll_gt_residual_mode=args.nll_gt_residual_mode,
        nll_gt_include_logdet_h=args.nll_gt_include_logdet_h,
        nll_vo_mode=args.nll_vo_mode,
        nll_mse_anchor_weight=args.nll_mse_anchor_weight,
        vo_residual_mode=args.vo_residual_mode,
        train_seqs=tuple(args.train_seqs),
        val_seqs=tuple(args.val_seqs),
        test_seqs=tuple(args.test_seqs),
    )


def load_vo_trajectories(dataset, args):
    if args.plato_supervision != "vo_pose":
        return None
    if args.vo_dir is None:
        raise ValueError("--vo_dir is required for --plato_supervision vo_pose")
    vo_by_seq = {}
    for index, seq in enumerate(dataset.sequences):
        vo_by_seq[seq] = load_vo_for_sequence(
            dataset,
            index,
            vo_dir=args.vo_dir,
            vo_template=args.vo_template,
            vo_align=args.vo_align,
            device=args.device,
        )
    return vo_by_seq


def run_test(dataset_test, args, gyro_checkpoint, acc_checkpoint):
    device = args.test_device
    bw_func = load_bw_func(args.network_type_Gyro, gyro_checkpoint, device)
    if gyro_checkpoint == acc_checkpoint:
        ba_func = ba_func_choose(args.network_type_Acc, device)
        checkpoint = torch.load(acc_checkpoint, map_location=device, weights_only=False)
        ba_func.load_state_dict(checkpoint["func_ba_model"])
    else:
        ba_func = load_ba_func(args.network_type_Acc, acc_checkpoint, device)
    with torch.no_grad():
        Test(
            dataset_test,
            args.outputdir,
            bw_func,
            ba_func,
            args.integral_method,
            device=device,
            recompute=args.test_recompute,
            bias_init_mode=args.test_bias_init_mode,
            bias_init_seconds=args.test_bias_init_seconds,
            mean_discretization=args.mean_discretization,
        )


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("true", "1", "yes", "y"):
        return True
    if lowered in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value")


if __name__ == "__main__":
    main()
