#!/usr/bin/env python3
"""Evaluate a preselected joint checkpoint on the five manuscript EuRoC sequences."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

from train_euroc_release import REPO, sha256, write_json
from check_paper_metrics import compare, REFERENCE_PATH
import torch
import numpy as np
from dataset import EUROCDataset, pload
from evalutation import AOE, ATE
from learning import Test
import mainEurocPLATO as euroc
from plato.training import load_ba_func, load_bw_func


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO / "external/Debias_IMU/data/EUROC")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--outputdir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bias-init", choices=["first", "zero"], default="first", help="first uses GT-derived b0 as in the research protocol; zero tests deployment-like initialization")
    args = parser.parse_args()
    torch.set_num_threads(1)
    outputdir = args.outputdir.resolve()
    if outputdir.exists() and any(outputdir.iterdir()):
        raise FileExistsError(f"Use an empty evaluation output directory: {outputdir}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "func_bw_model" not in checkpoint or "func_ba_model" not in checkpoint:
        raise ValueError("Expected a joint checkpoint with both bias networks")
    metadata = checkpoint.get("metadata", {})
    for name in ("network_type_Gyro", "network_type_Acc", "mean_discretization", "supervision"):
        if name not in metadata:
            raise ValueError(f"Checkpoint lacks required metadata: {name}")
    if metadata["supervision"] != "gt_se23":
        raise ValueError("This comparison requires GT-supervised training")
    held_out = set(euroc.DEFAULT_EUROC_TEST_SEQS)
    if not metadata.get("train_seqs") or not metadata.get("val_seqs"):
        raise ValueError("Checkpoint must declare training and validation sequences")
    if held_out & set([*metadata["train_seqs"], *metadata["val_seqs"]]):
        raise ValueError("Checkpoint trained or validated on a manuscript test sequence")
    cache = (args.cache_dir or outputdir / "cache").resolve()
    parsed = euroc.build_parser().parse_args(["--data_dir", str(args.data_dir.resolve()), "--data_cache_dir", str(cache), "--loss_window", "64"])
    parameters = euroc.euroc_dataset_parameters(parsed)
    dataset = EUROCDataset(**parameters, mode="test")
    bg = load_bw_func(metadata["network_type_Gyro"], str(args.checkpoint), args.device)
    ba = load_ba_func(metadata["network_type_Acc"], str(args.checkpoint), args.device)
    outputdir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "running", "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"), "training_metadata": metadata,
        "test_sequences": dataset.sequences, "bias_init": args.bias_init,
        "state_init": "first GT rotation, velocity and position", "alignment": "none",
        "mean_discretization": metadata["mean_discretization"],
        "torch": torch.__version__, "device": args.device, "threads": 1,
        "metrics": "AOE RMSE degrees; Euclidean position RMSE meters; equal sequence weighting",
        "data_sha256": {f"{seq}/{kind}": sha256(args.data_dir / seq / "mav0" / kind / "data.csv") for seq in dataset.sequences for kind in ("imu0", "state_groundtruth_estimate0")},
        "cache_sha256": {f"{seq}{suffix}": sha256(cache / (seq + suffix)) for seq in dataset.sequences for suffix in (".p", "_gt.p")},
    }
    manifest_path = outputdir / "evaluation_manifest.json"
    write_json(manifest_path, protocol)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            Test(dataset, str(outputdir), bg, ba, integral_method="euler", device=args.device, recompute=True, bias_init_mode=args.bias_init, compute_raw_int=False, mean_discretization=metadata["mean_discretization"])
        rows = []
        for sequence in dataset.sequences:
            result = pload(str(outputdir / "Sequnces-results" / sequence / "FullTra_result.p"))
            X_gt, solution = result["X_gt"], result["sol_pred"]
            imu_t = np.genfromtxt(args.data_dir / sequence / "mav0/imu0/data.csv", delimiter=",", skip_header=1, usecols=0)
            gt_t = np.genfromtxt(args.data_dir / sequence / "mav0/state_groundtruth_estimate0/data.csv", delimiter=",", skip_header=1, usecols=0)
            start = np.searchsorted(imu_t, max(imu_t[0], gt_t[0]))
            stop = np.searchsorted(imu_t, min(imu_t[-1], gt_t[-1]), side="right")
            overlap_duration = float((imu_t[stop - 1] - imu_t[start]) / 1e9)
            rows.append({"sequence": sequence, "duration_seconds": float(result["t_gt"][-1] - result["t_gt"][0]), "sample_count": len(result["t_gt"]), "input_overlap_duration_seconds": overlap_duration, "input_overlap_sample_count": int(stop - start), "aoe_deg": float(AOE(X_gt[:, :3, :3], result["R_pred"])), "position_rmse_m": float(ATE(X_gt[:, :3, 4], solution[:, 10:13])), "velocity_rmse_mps": float(ATE(X_gt[:, :3, 3], solution[:, 7:10]))})
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))["euroc_gt_preintegration"]
        comparison = compare(rows, reference, duration_policy="input_overlap")
        with (outputdir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(outputdir / "paper_comparison.json", comparison)
        protocol.update(status="completed", elapsed_seconds=time.perf_counter() - started, paper_comparison=comparison)
        write_json(manifest_path, protocol)
        print(json.dumps(comparison, indent=2), flush=True)
    except BaseException as error:
        protocol.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(manifest_path, protocol)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
