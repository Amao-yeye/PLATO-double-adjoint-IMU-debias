#!/usr/bin/env python3
"""Evaluate AquaLux on the complete five-sequence paper test set with GT validity masks."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

from train_euroc_release import REPO, sha256, write_json
from train_aqualux_release import PAPER_TEST_SEQUENCES
from check_paper_metrics import compare, REFERENCE_PATH
import torch
from aqualux_dataset import AquaLuxDataset, estimate_native_dt, read_imu_csv, read_tum
from dataset import pload, pdump
from evalutation import AOE, ATE
from learning import Test
from plato.training import load_ba_func, load_bw_func
from plato.inference import rollout_native_sequences
from network_for_BiadDy import VFSE23vbiasBase


def sequence_metrics(dataset, index: int, result: dict) -> dict:
    imu = dataset.load_imu(index)
    t = result["t_gt"]
    if len(t) != len(imu["t_imu"]) or not torch.equal(t.cpu(), imu["t_imu"].to(dtype=t.dtype).cpu()):
        raise ValueError("Evaluation must cover the complete prepared native-rate trajectory")
    nearest = imu["gt_nearest_dt"].to(result["X_gt"].device)
    valid = nearest <= dataset.gt_valid_max_dt
    if int(valid.sum()) < 10:
        raise ValueError("Too few valid GT samples for evaluation")
    invalid_mask = (~valid).long()
    X_gt, solution = result["X_gt"], result["sol_pred"]
    return {
        "sequence": dataset.sequences[index], "sample_count": len(t),
        "valid_metric_samples": int(valid.sum()), "duration_seconds": float(t[-1] - t[0]),
        "gt_valid_max_dt_seconds": dataset.gt_valid_max_dt,
        "aoe_deg": float(AOE(X_gt[:, :3, :3], result["R_pred"], mask=invalid_mask)),
        "position_rmse_m": float(ATE(X_gt[:, :3, 4], solution[:, 10:13], mask=invalid_mask)),
        "velocity_rmse_mps": float(ATE(X_gt[:, :3, 3], solution[:, 7:10], mask=invalid_mask)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--outputdir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bias-init", choices=["first", "zero"], default="first")
    parser.add_argument("--serial", action="store_true", help="Evaluate each sequence separately instead of batching the shared native grid")
    args = parser.parse_args()
    torch.set_num_threads(1)
    outputdir = args.outputdir.resolve()
    if outputdir.exists() and any(outputdir.iterdir()):
        raise FileExistsError(f"Use an empty evaluation directory: {outputdir}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    for key in ("func_bw_model", "func_ba_model"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint lacks {key}")
    for key in ("network_type_Gyro", "network_type_Acc", "mean_discretization", "supervision", "train_seqs", "val_seqs"):
        if key not in metadata:
            raise ValueError(f"Checkpoint lacks required metadata: {key}")
    if set(PAPER_TEST_SEQUENCES) & set([*metadata["train_seqs"], *metadata["val_seqs"]]):
        raise ValueError("Checkpoint training/validation overlaps with the held-out paper test set")
    if metadata["supervision"] == "gt_se23" and metadata.get("nll_gt_sup_stride") != 4:
        raise ValueError("GT NLL checkpoint must use stride 4")
    dt = estimate_native_dt(args.data_dir, PAPER_TEST_SEQUENCES)
    cache = args.cache_dir or outputdir / "cache"
    dataset = AquaLuxDataset(dataset_name="AquaLux", data_dir=str(args.data_dir), data_cache_dir=str(cache), train_seqs=[], val_seqs=[], test_seqs=PAPER_TEST_SEQUENCES, mode="test", percent_for_val=0.2, batch_size=1, loss_window=metadata.get("window_steps", 128), dt=dt, sg_window_bw=1, sg_order_bw=0, sg_window_ba=1, sg_order_ba=0)
    bg = load_bw_func(metadata["network_type_Gyro"], str(args.checkpoint), args.device)
    ba = load_ba_func(metadata["network_type_Acc"], str(args.checkpoint), args.device)
    outputdir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "running", "dataset": "AquaLux", "supervision": metadata["supervision"],
        "checkpoint_sha256": sha256(args.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"),
        "native_dt": dt, "native_rate_hz": 1/dt,
        "bias_init": args.bias_init, "state_init": "first GT R,v,p", "alignment": "none",
        "mean_discretization": metadata["mean_discretization"],
        "metric_mask": "retain samples with nearest original GT observation within 0.02 seconds",
        "aggregation": "unweighted mean of each sequence's RMSE on valid GT samples",
        "torch": torch.__version__, "device": args.device,
        "source_sha256": {str(path.relative_to(REPO)): sha256(path) for directory in (REPO / "external/Debias_IMU/BiasDy", REPO / "tools") for path in sorted(directory.rglob("*.py"))},
        "inference": "independent serial rollouts" if args.serial else "batched independent native-grid rollouts",
        "data_sha256": {f"{seq}/{name}": sha256(args.data_dir / "0524" / seq / relative) for seq in PAPER_TEST_SEQUENCES for name, relative in (("imu", "imu_diagnostics/imu_left.csv"), ("gt", "ATO/gt_imu_gravity_aligned.tum"))},
        "cache_sha256": {f"{seq}{suffix}": sha256(cache / (seq + suffix)) for seq in PAPER_TEST_SEQUENCES for suffix in (".p", "_gt.p")},
    }
    manifest_path = outputdir / "evaluation_manifest.json"
    write_json(manifest_path, protocol)
    started = time.perf_counter()
    try:
        if not args.serial and metadata["mean_discretization"] == "paper_zoh":
            model = VFSE23vbiasBase(biasfunc_w=bg, biasfunc_a=ba, device=args.device).to(args.device)
            model.set_gravity(dataset.gravity)
            model.eval()
            results = rollout_native_sequences(dataset, model, args.device, args.bias_init)
            for sequence, result in zip(dataset.sequences, results):
                directory = outputdir / "Sequnces-results" / sequence
                directory.mkdir(parents=True, exist_ok=True)
                pdump(result, str(directory / "FullTra_result.p"))
        else:
            with torch.no_grad():
                Test(dataset, str(outputdir), bg, ba, integral_method="euler", device=args.device, recompute=True, bias_init_mode=args.bias_init, compute_raw_int=False, mean_discretization=metadata["mean_discretization"])
        rows = []
        for index, sequence in enumerate(dataset.sequences):
            result = pload(str(outputdir / "Sequnces-results" / sequence / "FullTra_result.p"))
            row = sequence_metrics(dataset, index, result)
            imu_t, _ = read_imu_csv(args.data_dir / "0524" / sequence / "imu_diagnostics/imu_left.csv")
            gt_t, _, _ = read_tum(args.data_dir / "0524" / sequence / "ATO/gt_imu_gravity_aligned.tum")
            overlap = min(imu_t[-1], gt_t[-1]) - max(imu_t[0], gt_t[0])
            if abs(row["duration_seconds"] - overlap) > 2*dt:
                raise ValueError(f"Incomplete raw-input coverage for {sequence}")
            row["raw_overlap_duration_seconds"] = float(overlap)
            rows.append(row)
        with (outputdir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        references = json.loads(REFERENCE_PATH.read_text())
        comparison = compare(rows, references["aqualux_vo_preintegration"])
        comparison["comparison_condition"] = "GT-trained model vs paper VO-trained PLATO; supervision differs" if metadata["supervision"] == "gt_se23" else "VO-trained model vs paper VO-trained PLATO"
        if metadata["supervision"] == "gt_se23":
            comparison["paper_gt_hlcbd_comparison"] = compare(rows, references["aqualux_gt_hlcbd_preintegration"])
            comparison["paper_gt_hlcbd_comparison"]["comparison_condition"] = "GT-trained model vs paper GT-trained HL-CBD; model architectures differ"
        write_json(outputdir / "paper_comparison.json", comparison)
        protocol.update(status="completed", elapsed_seconds=time.perf_counter()-started, paper_comparison=comparison)
        write_json(manifest_path, protocol)
        print(json.dumps(comparison, indent=2), flush=True)
    except BaseException as error:
        protocol.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(manifest_path, protocol)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
