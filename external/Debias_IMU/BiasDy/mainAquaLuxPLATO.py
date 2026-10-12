#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch

BIASDY_DIR = os.path.abspath(os.path.dirname(__file__))
DEBIAS_ROOT = os.path.abspath(os.path.join(BIASDY_DIR, ".."))
if BIASDY_DIR not in sys.path:
    sys.path.insert(0, BIASDY_DIR)
if DEBIAS_ROOT not in sys.path:
    sys.path.insert(0, DEBIAS_ROOT)

from aqualux_dataset import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_GRAVITY_NORM,
    AquaLuxDataset,
    analyze_timing,
    estimate_native_dt,
    sorted_aqualux_sequences,
    write_timing_csv,
)
from learning import Test  # noqa: E402
from plato.aqualux_exports import (  # noqa: E402
    aggregate_rows,
    export_preintegration_results,
    write_metrics_csv as write_aqualux_metrics_csv,
)
from network_for_BiadDy import ba_func_choose  # noqa: E402
from plato.supervision import (  # noqa: E402
    build_vo_supervised_indices,
    load_vo_for_sequence,
    resolve_vo_supervision_stride,
)
from plato.training import (  # noqa: E402
    PlatoTrainConfig,
    load_ba_func,
    load_bw_func,
    train_mse_acc,
    train_mse_gyro,
    train_mse_joint,
    train_nll,
    train_nll_alternating,
)


DEFAULT_OUTPUT_DIR = "Results/AquaLux/preintegration/ours_gt_w64_se23_mse"
DEFAULT_TRAIN_SEQS = [
    "Train_000",
    "Train_002",
    "Train_008",
    "Train_010",
    "Train_012",
    "Train_014",
    "Train_016",
    "Train_018",
]
DEFAULT_TEST_SEQS = [f"Test_{idx:03d}" for idx in range(1, 20, 2)]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.test_device is None:
        args.test_device = args.device
    if args.loss_window is None:
        args.loss_window = args.plato_window_steps
    outputdir = Path(args.outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.data_cache_dir) if args.data_cache_dir else outputdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_seqs = args.train_seqs or DEFAULT_TRAIN_SEQS
    val_seqs = args.val_seqs or list(train_seqs)
    test_seqs = args.test_seqs or DEFAULT_TEST_SEQS
    all_seqs = sorted(set(train_seqs + val_seqs + test_seqs), key=sequence_sort_key)

    timing_rows = analyze_timing(args.data_dir, all_seqs)
    write_timing_csv(outputdir / "timing_report.csv", timing_rows)
    assert_no_bad_timing(timing_rows)
    dt = float(args.dt) if args.dt > 0 else estimate_native_dt(args.data_dir, all_seqs)

    dataset_parameters = {
        "dataset_name": "AquaLux",
        "data_dir": args.data_dir,
        "data_cache_dir": str(cache_dir),
        "train_seqs": train_seqs,
        "val_seqs": val_seqs,
        "test_seqs": test_seqs,
        "dt": dt,
        "percent_for_val": args.percent_for_val,
        "loss_window": args.loss_window,
        "batch_size": args.batch_size,
        "sg_window_bw": args.sg_window_bw,
        "sg_order_bw": args.sg_order_bw,
        "sg_window_ba": args.sg_window_ba,
        "sg_order_ba": args.sg_order_ba,
        "gravity_norm": args.gravity_norm,
        "gt_valid_max_dt": args.gt_valid_max_dt,
        "velocity_smooth_sec": args.velocity_smooth_sec,
    }
    dataset_train = AquaLuxDataset(**dataset_parameters, mode="train", recompute=args.recompute_data)
    dataset_val = AquaLuxDataset(**dataset_parameters, mode="val", recompute=False)
    dataset_test = AquaLuxDataset(**dataset_parameters, mode="test", recompute=False)

    config = make_train_config(args, train_seqs, val_seqs, test_seqs)
    vo_train = load_vo_trajectories(dataset_train, args)
    vo_val = load_vo_trajectories(dataset_val, args)
    if args.plato_supervision == "vo_rate_gt_se23":
        write_vo_rate_gt_window_audit(
            outputdir / "vo_rate_gt_window_audit.csv",
            dataset_train,
            vo_train,
            args,
        )

    write_run_config(outputdir / "run_config.json", args, train_seqs, val_seqs, test_seqs, dt, cache_dir)
    if args.cache_only:
        print(f"Cache-only run complete: {cache_dir}")
        return 0

    start = time.time()
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
            acc_checkpoint=acc_checkpoint,
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
        if gyro_checkpoint is None or acc_checkpoint is None:
            raise ValueError("--resume_gyro and --resume_acc are required for joint MSE")
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

    nll_checkpoint = None
    if args.plato_stage in ("nll", "both"):
        if gyro_checkpoint is None or acc_checkpoint is None:
            raise ValueError("--resume_gyro and --resume_acc are required for NLL")
        if args.nll_alternating:
            if any(
                (
                    args.nll_freeze_gyro,
                    args.nll_freeze_acc,
                    args.nll_freeze_covariance,
                    args.nll_covariance_only,
                )
            ):
                raise ValueError(
                    "alternating NLL cannot be combined with one-block freeze flags"
                )
            nll_checkpoint = train_nll_alternating(
                dataset_train,
                dataset_val,
                config,
                gyro_checkpoint=gyro_checkpoint,
                acc_checkpoint=acc_checkpoint,
                bias_lr=args.lr_NLL_bias,
                covariance_lr=args.lr_NLL_covariance,
                bias_weight_decay=args.weight_decay_NLL,
                covariance_weight_decay=args.weight_decay_NLL_covariance,
                epoch_count=args.epoch_nll,
                block_epochs=args.nll_block_epochs,
                start_with=args.nll_start_with,
                noise_checkpoint=args.resume_nll_noise,
                vo_train=vo_train,
                vo_val=vo_val,
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
                noise_checkpoint=args.resume_nll_noise,
                nll_freeze_covariance=args.nll_freeze_covariance,
                nll_covariance_only=args.nll_covariance_only,
                vo_train=vo_train,
                vo_val=vo_val,
            )

    elapsed = time.time() - start
    (outputdir / "time_memory_record.txt").write_text(
        f"Time for training: {elapsed}\n",
        encoding="utf-8",
    )

    if args.run_test:
        test_gyro = nll_checkpoint if nll_checkpoint is not None else gyro_checkpoint
        test_acc = nll_checkpoint if nll_checkpoint is not None else acc_checkpoint
        if test_gyro is None or (test_acc is None and not args.test_ba_zero):
            raise ValueError("--resume_gyro and --resume_acc are required for --run_test unless --test_ba_zero is set")
        run_test(dataset_test, args, test_gyro, test_acc)
        rows = export_preintegration_results(dataset_test, outputdir)
        write_aqualux_metrics_csv(outputdir / "metrics.csv", rows)
        aggregate = aggregate_rows(rows)
        write_plato_method_summary(outputdir / "summary.md", rows, aggregate, args, dt)
        write_improvement_summary(outputdir / "target_summary.md", outputdir, rows)
        print(
            f"AquaLux PLATO aggregate: position_rmse_m={aggregate['position_rmse_m']:.6f}, "
            f"rotation_rmse_deg={aggregate['rotation_rmse_deg']:.6f}"
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("PLATO AquaLux training for Debias_IMU")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--data_cache_dir", type=str, default=None)
    parser.add_argument("--outputdir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--integral_method", choices=["euler"], default="euler")
    parser.add_argument(
        "--mean_discretization",
        choices=["paper_zoh"],
        default="paper_zoh",
        help=(
            "Structure-preserving ZOH state propagation with Euler bias dynamics."
        ),
    )
    parser.add_argument("--network_type_Gyro", type=str, default="bw_func_net")
    parser.add_argument("--network_type_Acc", type=str, default="ba_func_net_rate_tanh_1e2")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--test_device", type=str, default=None)
    parser.add_argument("--recompute_data", action="store_true")
    parser.add_argument("--cache_only", action="store_true")

    parser.add_argument("--train_seqs", nargs="+", default=None)
    parser.add_argument("--val_seqs", nargs="+", default=None)
    parser.add_argument("--test_seqs", nargs="+", default=None)
    parser.add_argument("--dt", type=float, default=0.0)
    parser.add_argument("--percent_for_val", type=float, default=0.2)
    parser.add_argument("--loss_window", type=int, default=None)
    parser.add_argument("--plato_window_steps", type=int, default=64)
    parser.add_argument(
        "--plato_supervision",
        choices=["gt_se23", "vo_rate_gt_se23", "vo_pose", "vo_se23"],
        default="gt_se23",
    )
    parser.add_argument("--plato_stage", choices=["gyro", "mse", "acc", "joint", "nll", "both", "eval"], default="acc")
    parser.add_argument("--plato_sup_stride", type=int, default=None)
    parser.add_argument("--plato_sup_rate_hz", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=1000)

    parser.add_argument("--epoch_mse", type=int, default=1801)
    parser.add_argument("--epoch_joint_mse", type=int, default=0)
    parser.add_argument("--epoch_nll", type=int, default=300)
    parser.add_argument("--lr_Gyro", type=float, default=1e-6)
    parser.add_argument("--lr_Acc", type=float, default=1e-6)
    parser.add_argument("--lr_joint_mse", type=float, default=1e-6)
    parser.add_argument("--lr_NLL", type=float, default=1e-6)
    parser.add_argument("--lr_NLL_bias", type=float, default=2e-7)
    parser.add_argument("--lr_NLL_covariance", type=float, default=5e-5)
    parser.add_argument("--weight_decay_Gyro", type=float, default=1e-6)
    parser.add_argument("--weight_decay_Acc", type=float, default=1e-6)
    parser.add_argument("--weight_decay_joint_mse", type=float, default=1e-6)
    parser.add_argument("--weight_decay_NLL", type=float, default=1e-6)
    parser.add_argument("--weight_decay_NLL_covariance", type=float, default=0.0)
    parser.add_argument("--val_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--train_sequence_strategy", choices=["random", "cycle"], default="random")

    parser.add_argument("--mse_rot_weight", type=float, default=1.0)
    parser.add_argument("--mse_vel_weight", type=float, default=1.0)
    parser.add_argument("--mse_pos_weight", type=float, default=10.0)
    parser.add_argument("--mse_acc_rot_weight", type=float, default=0.0)
    parser.add_argument("--mse_acc_vel_weight", type=float, default=1.0)
    parser.add_argument("--mse_acc_pos_weight", type=float, default=10.0)
    parser.add_argument("--mse_gyro_full_state", action="store_true")
    parser.add_argument("--gyro_train_with_acc", action="store_true")
    parser.add_argument("--mse_residual_mode", choices=["log", "right", "global", "hybrid"], default="right")
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
    parser.add_argument("--resume_acc", type=str, default=None)
    parser.add_argument("--vo_dir", type=str, default=None)
    parser.add_argument("--vo_template", type=str, default="{seq}/ATO/vo_imu_gravity_aligned.tum")
    parser.add_argument("--vo_align", choices=["first_gt", "none"], default="first_gt")
    parser.add_argument("--vo_residual_mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--vo_init_mode", choices=["gt", "vo_zero_bias"], default="gt")

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
        "--nll_gt_blocks",
        choices=["full", "rot"],
        default="full",
        help=(
            "Use the full SE_2(3) sparse GT likelihood or its exact autonomous "
            "rotation marginal."
        ),
    )
    parser.add_argument("--nll_gt_residual_mode", choices=["right", "log", "hybrid_global"], default="right")
    parser.add_argument(
        "--nll_gt_include_logdet_h",
        type=str2bool,
        default=True,
        help=(
            "Include Eq. (9) sum_i log|det H_i| in sparse right-residual GT "
            "NLL. Set false only for the controlled ablation."
        ),
    )
    parser.add_argument("--nll_vo_mode", choices=["woodbury", "sparse"], default="sparse")
    parser.add_argument("--nll_mse_anchor_weight", type=float, default=0.001)
    parser.add_argument("--nll_freeze_gyro", action="store_true")
    parser.add_argument("--nll_freeze_acc", action="store_true")
    parser.add_argument("--resume_nll_noise", type=str, default=None)
    parser.add_argument("--nll_freeze_covariance", action="store_true")
    parser.add_argument("--nll_covariance_only", action="store_true")
    parser.add_argument("--nll_alternating", action="store_true")
    parser.add_argument("--nll_block_epochs", type=int, default=30)
    parser.add_argument(
        "--nll_start_with", choices=["bias", "covariance"], default="bias"
    )

    parser.add_argument("--run_test", action="store_true")
    parser.add_argument("--test_recompute", action="store_true")
    parser.add_argument("--test_skip_raw_int", action="store_true")
    parser.add_argument("--test_bias_init_mode", choices=["zero", "first", "mean"], default="first")
    parser.add_argument("--test_bias_init_seconds", type=float, default=5.0)
    parser.add_argument("--test_ba_zero", action="store_true")
    parser.add_argument("--gravity_norm", type=float, default=DEFAULT_GRAVITY_NORM)
    parser.add_argument("--gt_valid_max_dt", type=float, default=0.02)
    parser.add_argument("--velocity_smooth_sec", type=float, default=0.12)
    parser.add_argument("--sg_window_bw", type=int, default=1)
    parser.add_argument("--sg_order_bw", type=int, default=0)
    parser.add_argument("--sg_window_ba", type=int, default=1)
    parser.add_argument("--sg_order_ba", type=int, default=0)
    return parser


def make_train_config(args, train_seqs: list[str], val_seqs: list[str], test_seqs: list[str]) -> PlatoTrainConfig:
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
        mse_gyro_full_state=args.mse_gyro_full_state,
        gyro_train_with_acc=args.gyro_train_with_acc,
        mse_residual_mode=args.mse_residual_mode,
        gradient_mode=args.gradient_mode,
        covariance_gradient_mode=args.covariance_gradient_mode,
        analytic_grad_scale=args.analytic_grad_scale,
        analytic_grad_clip_norm=args.analytic_grad_clip_norm,
        analytic_residual_jacobian=args.analytic_residual_jacobian,
        gyro_multi_window_steps=tuple(args.gyro_multi_window_steps or ()),
        gyro_multi_window_weights=tuple(args.gyro_multi_window_weights or ()),
        nll_gt_mode=args.nll_gt_mode,
        nll_gt_blocks=args.nll_gt_blocks,
        nll_gt_residual_mode=args.nll_gt_residual_mode,
        nll_gt_include_logdet_h=args.nll_gt_include_logdet_h,
        nll_vo_mode=args.nll_vo_mode,
        nll_mse_anchor_weight=args.nll_mse_anchor_weight,
        vo_residual_mode=args.vo_residual_mode,
        vo_init_mode=args.vo_init_mode,
        train_seqs=tuple(train_seqs),
        val_seqs=tuple(val_seqs),
        test_seqs=tuple(test_seqs),
    )


def load_vo_trajectories(dataset: AquaLuxDataset, args) -> dict | None:
    if args.plato_supervision not in (
        "vo_rate_gt_se23",
        "vo_pose",
        "vo_se23",
    ):
        return None
    vo_dir = args.vo_dir or str(Path(args.data_dir) / "0524")
    vo_by_seq = {}
    for index, seq in enumerate(dataset.sequences):
        vo_align = (
            "none"
            if args.plato_supervision == "vo_rate_gt_se23"
            else args.vo_align
        )
        vo_by_seq[seq] = load_vo_for_sequence(
            dataset,
            index,
            vo_dir=vo_dir,
            vo_template=args.vo_template,
            vo_align=vo_align,
            device=args.device,
        )
    return vo_by_seq


def write_vo_rate_gt_window_audit(
    path: Path,
    dataset: AquaLuxDataset,
    trajectories: dict,
    args,
) -> None:
    """Record the exact dense-IMU/64-GT geometry used by the control."""
    fields = [
        "sequence",
        "vo_samples",
        "vo_median_dt_sec",
        "vo_median_rate_hz",
        "native_imu_dt_sec",
        "resolved_imu_stride",
        "supervised_gt_states",
        "dense_imu_states",
        "rollout_duration_sec",
        "target_source",
        "vo_pose_values_used",
    ]
    rows = []
    for sequence in dataset.sequences:
        trajectory = trajectories[sequence]
        positive_dt = trajectory.t[1:] - trajectory.t[:-1]
        positive_dt = positive_dt[positive_dt > 0]
        if positive_dt.numel() == 0:
            raise ValueError(f"{sequence}: VO timestamps must increase")
        median_dt = float(torch.median(positive_dt).item())
        stride = resolve_vo_supervision_stride(
            dataset.dt,
            trajectory,
            sup_stride=args.plato_sup_stride,
            sup_rate_hz=args.plato_sup_rate_hz,
        )
        dense_steps, supervised_indices = build_vo_supervised_indices(
            args.plato_window_steps,
            stride,
        )
        rows.append(
            {
                "sequence": sequence,
                "vo_samples": int(trajectory.t.numel()),
                "vo_median_dt_sec": f"{median_dt:.12g}",
                "vo_median_rate_hz": f"{1.0 / median_dt:.12g}",
                "native_imu_dt_sec": f"{dataset.dt:.12g}",
                "resolved_imu_stride": int(stride),
                "supervised_gt_states": int(supervised_indices.numel()),
                "dense_imu_states": int(dense_steps),
                "rollout_duration_sec": f"{(dense_steps - 1) * dataset.dt:.12g}",
                "target_source": "GT",
                "vo_pose_values_used": False,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_test(dataset_test: AquaLuxDataset, args, gyro_checkpoint: str, acc_checkpoint: str | None) -> None:
    device = args.test_device
    bw_func = load_bw_func(args.network_type_Gyro, gyro_checkpoint, device)
    if args.test_ba_zero:
        ba_func = ba_func_choose(args.network_type_Acc, device)
    elif gyro_checkpoint == acc_checkpoint:
        checkpoint = torch.load(acc_checkpoint, map_location=device, weights_only=False)
        ba_func = ba_func_choose(args.network_type_Acc, device)
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
            compute_raw_int=not args.test_skip_raw_int,
            disable_acc_bias=args.test_ba_zero,
            mean_discretization=args.mean_discretization,
        )


def write_run_config(
    path: Path,
    args,
    train_seqs: list[str],
    val_seqs: list[str],
    test_seqs: list[str],
    dt: float,
    cache_dir: Path,
) -> None:
    payload = vars(args).copy()
    payload.update(
        {
            "method": Path(args.outputdir).name,
            "dataset": "AquaLux",
            "train_seqs": train_seqs,
            "val_seqs": val_seqs,
            "test_seqs": test_seqs,
            "effective_dt": dt,
            "effective_rate_hz": 1.0 / dt,
            "cache_dir": str(cache_dir),
            "strict_se23_residual": args.mse_residual_mode == "right",
            "nll_dtype": "float64",
            "gt_nll_dense_supervision_stride": args.nll_gt_sup_stride,
            "supervision_target_source": "GT",
            "vo_pose_values_used": args.plato_supervision in ("vo_pose", "vo_se23"),
            "window_semantics": (
                "W counts VO-rate GT supervised states; native IMU rollout is dense"
                if args.plato_supervision == "vo_rate_gt_se23"
                else "W counts native-IMU or VO supervision states according to mode"
            ),
            "axb_used": False,
        }
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_improvement_summary(path: Path, outputdir: Path, rows: list[dict[str, float | int | str]]) -> None:
    baseline_path = outputdir.parent / "rss_baseline_unseen_corrected" / "metrics.csv"
    if not baseline_path.exists():
        return
    import csv
    import numpy as np

    with baseline_path.open("r", encoding="utf-8", newline="") as handle:
        baseline_rows = {row["sequence"]: row for row in csv.DictReader(handle)}

    method_sequences = [str(row["sequence"]) for row in rows]
    missing = [seq for seq in method_sequences if seq not in baseline_rows]
    if missing:
        raise KeyError(f"Missing baseline metrics for sequences: {missing}")
    matched_baseline_rows = [baseline_rows[seq] for seq in method_sequences]
    mean_pos = float(np.mean([float(row["position_rmse_m"]) for row in rows]))
    mean_rot = float(np.mean([float(row["rotation_rmse_deg"]) for row in rows]))
    base_mean_pos = float(np.mean([float(row["position_rmse_m"]) for row in matched_baseline_rows]))
    base_mean_rot = float(np.mean([float(row["rotation_rmse_deg"]) for row in matched_baseline_rows]))
    pos_improve = improvement(base_mean_pos, mean_pos)
    rot_improve = improvement(base_mean_rot, mean_rot)

    lines = [
        "# AquaLux PLATO Target Summary",
        "",
        f"- Baseline mean position RMSE: `{base_mean_pos:.6f} m`.",
        f"- Baseline mean rotation RMSE: `{base_mean_rot:.6f} deg`.",
        f"- Method mean position RMSE: `{mean_pos:.6f} m`.",
        f"- Method mean rotation RMSE: `{mean_rot:.6f} deg`.",
        f"- Position improvement: `{pos_improve:.6f}%`.",
        f"- Rotation improvement: `{rot_improve:.6f}%`.",
        f"- GT-MSE target pass: `{pos_improve >= 8.0 and rot_improve >= 5.0}`.",
        "",
        "| sequence | baseline pos | method pos | pos improve | baseline rot | method rot | rot improve |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        seq = str(row["sequence"])
        base = baseline_rows[seq]
        bp = float(base["position_rmse_m"])
        br = float(base["rotation_rmse_deg"])
        mp = float(row["position_rmse_m"])
        mr = float(row["rotation_rmse_deg"])
        lines.append(
            f"| `{seq}` | {bp:.6f} | {mp:.6f} | {improvement(bp, mp):.6f}% | "
            f"{br:.6f} | {mr:.6f} | {improvement(br, mr):.6f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plato_method_summary(
    path: Path,
    rows: list[dict[str, float | int | str]],
    aggregate: dict[str, float | int],
    args,
    dt: float,
) -> None:
    mean_pos = sum(float(row["position_rmse_m"]) for row in rows) / len(rows)
    mean_rot = sum(float(row["rotation_rmse_deg"]) for row in rows) / len(rows)
    method = Path(args.outputdir).name
    lines = [
        "# AquaLux PLATO Preintegration Summary",
        "",
        "## Scope",
        "",
        f"- Method: `{method}`.",
        f"- Supervision: `{args.plato_supervision}`.",
        f"- Stage: `{args.plato_stage}`.",
        f"- Gyro network: `{args.network_type_Gyro}`.",
        f"- Acc network: `{args.network_type_Acc}`.",
        f"- Resume gyro checkpoint: `{args.resume_gyro}`.",
        f"- Resume acc checkpoint: `{args.resume_acc}`.",
        f"- Gyro full-state loss: `{getattr(args, 'mse_gyro_full_state', False)}`.",
        f"- Gyro training with frozen acc: `{getattr(args, 'gyro_train_with_acc', False)}`.",
        f"- Strict SE_2(3) residual: `{args.mse_residual_mode}`.",
        f"- Loss window: `{args.loss_window}` samples.",
        f"- Effective IMU rate: `{1.0 / dt:.3f} Hz` (`dt={dt:.12g} s`).",
        f"- Gravity: `[0, 0, -{args.gravity_norm:g}] m/s^2`.",
        f"- Valid metric samples require nearest source GT pose within `{args.gt_valid_max_dt:g} s`.",
        f"- Train sequences: `{', '.join(args.train_seqs or DEFAULT_TRAIN_SEQS)}`.",
        f"- Test sequences: `{', '.join(args.test_seqs or DEFAULT_TEST_SEQS)}`.",
        "",
        "## Aggregate",
        "",
        "| Method | Sequences | Valid samples | Pooled pos RMSE (m) | Pooled rot RMSE (deg) | Mean seq pos RMSE (m) | Mean seq rot RMSE (deg) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| `{method}` | {len(rows)} | {aggregate['valid_metric_samples']} | "
            f"{aggregate['position_rmse_m']:.6f} | {aggregate['rotation_rmse_deg']:.6f} | "
            f"{mean_pos:.6f} | {mean_rot:.6f} |"
        ),
    ]
    if args.plato_stage == "acc" and args.resume_gyro:
        lines.extend(
            [
                "",
                "Note: this run freezes the gyro network, so rotation-only metrics are expected",
                "to match the resumed gyro checkpoint unless the acc-stage changes position",
                "alignment side effects in a downstream evaluator.",
            ]
        )
    lines.extend(
        [
            "",
            "## Per-Sequence Results",
            "",
            "| Sequence | Samples | Valid samples | Duration (s) | GT max nearest dt (s) | Position RMSE (m) | Rotation RMSE (deg) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['sequence']}` | {row['samples']} | {row['valid_metric_samples']} | "
            f"{float(row['duration_sec']):.3f} | {float(row['gt_interp_max_dt_sec']):.6f} | "
            f"{float(row['position_rmse_m']):.6f} | {float(row['rotation_rmse_deg']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Output directory: `{path.parent}`.",
            "- Per-sequence files:",
            "  - `gt_resampled.tum`",
            "  - `bias_dynamic_preintegrated.tum`",
            "  - `corrected_imu.csv`",
            "  - `errors.csv`",
            "- Machine-readable metrics: `metrics.csv`.",
            "- Target comparison: `target_summary.md`.",
            "- Timing audit: `timing_report.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_no_bad_timing(rows: list[dict[str, float | int | str]]) -> None:
    bad = [
        row["sequence"]
        for row in rows
        if int(row["imu_duplicate_dt"]) or int(row["imu_time_backjump"]) or int(row["gt_duplicate_dt"]) or int(row["gt_time_backjump"])
    ]
    if bad:
        raise RuntimeError(f"AquaLux time audit failed for sequences: {bad}")


def improvement(baseline: float, method: float) -> float:
    return (baseline - method) / baseline * 100.0 if baseline != 0 else float("nan")


def sequence_sort_key(name: str) -> tuple[str, int]:
    prefix, number = name.split("_", 1)
    return prefix, int(number)


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
    raise SystemExit(main())
