from __future__ import annotations

import argparse
import csv
import os
import sys

import torch

BIASDY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEBIAS_ROOT = os.path.abspath(os.path.join(BIASDY_DIR, ".."))
if BIASDY_DIR not in sys.path:
    sys.path.insert(0, BIASDY_DIR)
if DEBIAS_ROOT not in sys.path:
    sys.path.insert(0, DEBIAS_ROOT)

import Interpolation
from dataset import EUROCDataset
from evalutation import AOE
from network_for_BiadDy import VFSE23vbiasBase
from SO3diffeq import odeint_SO3
from plato.training import load_ba_func, load_bw_func


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
    parser = argparse.ArgumentParser("Fast gyro AOE evaluation for PLATO checkpoints")
    parser.add_argument("--resume_gyro", required=True)
    parser.add_argument("--resume_acc", required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--network_type_Gyro", default="bw_func_net")
    parser.add_argument("--network_type_Acc", default="ba_func_net")
    parser.add_argument("--integral_method", default="euler")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_dir", default="data/EUROC")
    parser.add_argument("--data_cache_dir", default="data/EUROC")
    parser.add_argument("--loss_window", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--test_seqs", nargs="+", default=DEFAULT_EUROC_TEST_SEQS)
    args = parser.parse_args()

    dataset = EUROCDataset(
        dataset_name="EUROC",
        data_dir=args.data_dir,
        data_cache_dir=args.data_cache_dir,
        train_seqs=DEFAULT_EUROC_TRAIN_VAL_SEQS,
        val_seqs=DEFAULT_EUROC_TRAIN_VAL_SEQS,
        test_seqs=args.test_seqs,
        mode="test",
        dt=0.005,
        percent_for_val=0.2,
        loss_window=args.loss_window,
        batch_size=args.batch_size,
        sg_window_bw=1,
        sg_order_bw=0,
        sg_window_ba=1,
        sg_order_ba=0,
    )

    device = args.device
    bw_func = load_bw_func(args.network_type_Gyro, args.resume_gyro, device)
    ba_func = load_ba_func(args.network_type_Acc, args.resume_acc, device)
    model = VFSE23vbiasBase(biasfunc_w=bw_func, biasfunc_a=ba_func, device=device).to(device)
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    model.eval()

    rows = []
    with torch.no_grad():
        for index, seq in enumerate(dataset.sequences):
            coeff, time = dataset.get_coeff(index)
            spline = Interpolation.CubicHermiteSpline(time, coeff, device)
            _, _, X_gt, t_gt = dataset.get_full_trajectory(index, device=device)
            R0 = X_gt[0, :3, :3].clone()
            bias_w_gt, _ = dataset.get_w_bias(index)
            bias_a_gt, _ = dataset.get_a_bias(index)
            bw0 = bias_w_gt[0].to(device)
            ba0 = bias_a_gt[0].to(device)
            vec3zeros = torch.zeros(3, device=device)
            y0 = torch.cat(
                [vec3zeros, t_gt[0].unsqueeze(-1), bw0, X_gt[0, :3, 3], X_gt[0, :3, 4], ba0],
                dim=-1,
            ).unsqueeze(0)

            model.set_u_func(spline.evaluate)
            model.set_u_dot_func(spline.derivative)
            model.set_R0(R0)
            _, R_pred = odeint_SO3(model, y0, R0, t_gt, method=args.integral_method)
            R_pred = R_pred.squeeze(1)
            aoe = float(AOE(X_gt[:, :3, :3], R_pred).detach().cpu())
            rows.append({"sequence": seq, "plato_aoe_deg": aoe})
            print(f"Seq {seq:16} | PLATO AOE {aoe:.6f} deg")

    mean_aoe = sum(row["plato_aoe_deg"] for row in rows) / len(rows)
    print(f"Mean AOE {mean_aoe:.10f} deg")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sequence", "plato_aoe_deg"])
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({"sequence": "mean", "plato_aoe_deg": mean_aoe})


if __name__ == "__main__":
    main()
