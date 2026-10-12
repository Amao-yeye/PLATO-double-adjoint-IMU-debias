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

from dataset import pload
from evalutation import AOE, ATE
from plato.supervision import PoseTrajectory, align_vo_first_gt, interpolate_pose, read_vo_pose_file


EUROC_TRAIN_VAL_SEQUENCES = [
    "MH_01_easy",
    "MH_03_medium",
    "MH_05_difficult",
    "V1_02_medium",
    "V2_01_easy",
    "V2_03_difficult",
]


def evaluate_vo_sequence(
    seq: str,
    gt_cache_dir: str,
    vo_dir: str,
    vo_template: str,
    vo_align: str = "first_gt",
    device: str = "cpu",
) -> dict[str, float | int | str]:
    imudict = pload(gt_cache_dir, seq + ".p")
    gtdict = pload(gt_cache_dir, seq + "_gt.p")
    vo_path = os.path.join(vo_dir, vo_template.format(seq=seq))
    vo_traj = read_vo_pose_file(
        vo_path,
        sequence_start_timestamp=imudict["t_start"],
        device=device,
        dtype=torch.float32,
    )
    gt_traj = PoseTrajectory(
        t=gtdict["t_gt"].to(device=device, dtype=torch.float32),
        R=gtdict["R_gt"].to(device=device, dtype=torch.float32),
        p=gtdict["p_gt"].to(device=device, dtype=torch.float32),
    )
    if vo_align == "first_gt":
        vo_traj = align_vo_first_gt(vo_traj, gt_traj)
    elif vo_align != "none":
        raise ValueError("vo_align must be one of: first_gt, none")

    valid = (gt_traj.t >= vo_traj.t[0]) & (gt_traj.t <= vo_traj.t[-1])
    if valid.sum() < 2:
        raise ValueError(f"No sufficient GT/VO overlap for {seq}")

    t_eval = gt_traj.t[valid]
    R_gt = gt_traj.R[valid]
    p_gt = gt_traj.p[valid]
    R_vo, p_vo = interpolate_pose(t_eval, vo_traj)

    pos_rmse = ATE(p_gt, p_vo)
    rot_rmse = AOE(R_gt, R_vo)
    return {
        "sequence": seq,
        "samples": int(t_eval.numel()),
        "duration_s": float(t_eval[-1] - t_eval[0]),
        "vo_start_s": float(vo_traj.t[0]),
        "vo_end_s": float(vo_traj.t[-1]),
        "pos_rmse_m": float(pos_rmse.detach().cpu()),
        "rot_rmse_deg": float(rot_rmse.detach().cpu()),
        "vo_path": vo_path,
    }


def evaluate_vo_quality(args) -> list[dict[str, float | int | str]]:
    rows = []
    for seq in args.sequences:
        rows.append(
            evaluate_vo_sequence(
                seq=seq,
                gt_cache_dir=args.gt_cache_dir,
                vo_dir=args.vo_dir,
                vo_template=args.vo_template,
                vo_align=args.vo_align,
                device=args.device,
            )
        )
    return rows


def write_csv(rows: list[dict[str, float | int | str]], output_csv: str) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        "sequence",
        "samples",
        "duration_s",
        "vo_start_s",
        "vo_end_s",
        "pos_rmse_m",
        "rot_rmse_deg",
        "vo_path",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, float | int | str]]) -> None:
    print("sequence,samples,duration_s,pos_rmse_m,rot_rmse_deg")
    for row in rows:
        print(
            f"{row['sequence']},{row['samples']},"
            f"{row['duration_s']:.3f},{row['pos_rmse_m']:.6f},{row['rot_rmse_deg']:.6f}"
        )
    if rows:
        pos_mean = sum(float(row["pos_rmse_m"]) for row in rows) / len(rows)
        rot_mean = sum(float(row["rot_rmse_deg"]) for row in rows) / len(rows)
        print(f"mean,,,{pos_mean:.6f},{rot_mean:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Evaluate VO pose quality against cached EuRoC GT")
    parser.add_argument("--gt_cache_dir", type=str, default="data/EUROC")
    parser.add_argument("--vo_dir", type=str, required=True)
    parser.add_argument("--vo_template", type=str, default="Euroc_VO/{seq}/data.csv")
    parser.add_argument("--vo_align", choices=["first_gt", "none"], default="first_gt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_csv", type=str, default="results/vo_quality/vo_gt_rmse.csv")
    parser.add_argument("--sequences", nargs="+", default=EUROC_TRAIN_VAL_SEQUENCES)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = evaluate_vo_quality(args)
    write_csv(rows, args.output_csv)
    print_table(rows)
    print(f"Wrote VO quality CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
