#!/usr/bin/env python3
"""Train native-rate GT MSE and stride-4 GT NLL on AquaLux from raw data."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from train_euroc_release import REPO, BIASDY, sha256, write_json
import torch
import mainAquaLuxPLATO as aqua
from aqualux_dataset import AquaLuxDataset, analyze_timing, estimate_native_dt
from plato.training import train_mse_gyro, train_mse_acc, train_nll_alternating

PAPER_TEST_SEQUENCES = ["Test_001", "Test_005", "Test_007", "Test_017", "Test_019"]


def resolve_recipe(recipe: dict, data_dir: Path, outputdir: Path, device: str):
    allowed = {"dataset", "gyro_updates", "acc_updates", "nll_updates", "checkpoint_selection", "parameters"}
    if set(recipe) - allowed or recipe.get("dataset") != "aqualux" or recipe.get("checkpoint_selection") != "final":
        raise ValueError("Expected an AquaLux recipe with predeclared final-step selection")
    for key in ("gyro_updates", "acc_updates", "nll_updates"):
        if type(recipe.get(key)) is not int or recipe[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    parser = aqua.build_parser()
    actions = {action.dest: action for action in parser._actions}
    controlled = {"data_dir", "data_cache_dir", "outputdir", "device", "test_device", "resume_gyro", "resume_acc", "run_test", "cache_only", "plato_stage", "recompute_data", "train_seqs", "val_seqs", "test_seqs", "epoch_mse", "epoch_nll", "nll_alternating", "dt", "gravity_norm", "gt_valid_max_dt", "velocity_smooth_sec", "sg_window_bw", "sg_order_bw", "sg_window_ba", "sg_order_ba"}
    argv = []
    for name, value in recipe["parameters"].items():
        if name not in actions or name in controlled:
            raise ValueError(f"Unsupported recipe parameter: {name}")
        action = actions[name]
        if isinstance(action, argparse._StoreTrueAction):
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean")
            if value:
                argv.append(action.option_strings[0])
        else:
            argv.append(action.option_strings[0])
            argv.extend(str(item) for item in value) if isinstance(value, list) else argv.append(str(value))
    args = parser.parse_args(argv)
    args.data_dir, args.outputdir = str(data_dir), str(outputdir)
    args.device = device
    args.loss_window = args.plato_window_steps
    if args.plato_supervision != "gt_se23" or args.plato_sup_stride is not None or args.plato_sup_rate_hz is not None:
        raise ValueError("This recipe requires native-IMU-rate GT MSE")
    if args.nll_gt_sup_stride != 4:
        raise ValueError("GT NLL requires stride 4")
    if args.gradient_mode != "double_adjoint" or args.covariance_gradient_mode != "forward_sensitivity":
        raise ValueError("Alternating NLL requires double adjoint and forward covariance sensitivity")
    config = aqua.make_train_config(args, aqua.DEFAULT_TRAIN_SEQS, aqua.DEFAULT_TRAIN_SEQS, PAPER_TEST_SEQUENCES)
    return args, config


def dataset_parameters(args, cache: Path, dt: float) -> dict:
    return {
        "dataset_name": "AquaLux", "data_dir": args.data_dir, "data_cache_dir": str(cache),
        "train_seqs": aqua.DEFAULT_TRAIN_SEQS, "val_seqs": aqua.DEFAULT_TRAIN_SEQS,
        "test_seqs": PAPER_TEST_SEQUENCES, "dt": dt, "percent_for_val": args.percent_for_val,
        "loss_window": args.loss_window, "batch_size": args.batch_size,
        "sg_window_bw": args.sg_window_bw, "sg_order_bw": args.sg_order_bw,
        "sg_window_ba": args.sg_window_ba, "sg_order_ba": args.sg_order_ba,
        "gravity_norm": args.gravity_norm, "gt_valid_max_dt": args.gt_valid_max_dt,
        "velocity_smooth_sec": args.velocity_smooth_sec,
    }


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/aqualux_gt_release.json")
    parser.add_argument("--data-dir", type=Path, required=True, help="AquaLux 0524_data directory")
    parser.add_argument("--outputdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-test", action="store_true")
    options = parser.parse_args()
    torch.set_num_threads(1)
    outputdir, data_dir = options.outputdir.resolve(), options.data_dir.resolve()
    recipe = json.loads(options.config.read_text())
    args, config = resolve_recipe(recipe, data_dir, outputdir, options.device)
    if outputdir.exists() and any(outputdir.iterdir()):
        raise FileExistsError(f"Use an empty experiment directory: {outputdir}")
    sequences = list(dict.fromkeys(aqua.DEFAULT_TRAIN_SEQS + PAPER_TEST_SEQUENCES))
    timing = analyze_timing(data_dir, sequences)
    aqua.assert_no_bad_timing(timing)
    dt = estimate_native_dt(data_dir, sequences)
    outputdir.mkdir(parents=True, exist_ok=True)
    aqua.write_timing_csv(outputdir / "timing_report.csv", timing)
    manifest = {
        "status": "running", "recipe": recipe, "data_source": "raw IMU CSV and GT TUM",
        "native_dt": dt, "native_rate_hz": 1 / dt,
        "gt_mse_sampling": "GT interpolated onto every native IMU state; no thinning",
        "gt_nll_sampling": "retain indices 0,4,8,...; propagate every native IMU interval",
        "preprocessing": {"gravity_norm": args.gravity_norm, "gt_valid_max_dt": args.gt_valid_max_dt, "velocity_smooth_sec": args.velocity_smooth_sec, "bias_smoothing": "none (window 1)"},
        "train_sequences": aqua.DEFAULT_TRAIN_SEQS, "test_sequences": PAPER_TEST_SEQUENCES,
        "validation": "last 20 percent of each training sequence", "seed_per_stage": 3407,
        "checkpoint_selection": "predeclared final optimizer steps; tests do not select checkpoints",
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": options.device, "gpu": torch.cuda.get_device_name(options.device) if options.device.startswith("cuda") else None,
        "data_sha256": {f"{seq}/{name}": sha256(data_dir / "0524" / seq / relative) for seq in sequences for name, relative in (("imu", "imu_diagnostics/imu_left.csv"), ("gt", "ATO/gt_imu_gravity_aligned.tum"))},
        "source_sha256": {str(path.relative_to(REPO)): sha256(path) for base in (BIASDY, REPO / "tools") for path in sorted(base.rglob("*.py"))},
    }
    manifest_path = outputdir / "run_manifest.json"
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    try:
        parameters = dataset_parameters(args, outputdir / "cache", dt)
        train = AquaLuxDataset(**parameters, mode="train")
        val = AquaLuxDataset(**parameters, mode="val")
        bg = train_mse_gyro(train, val, replace(config, outputdir=str(outputdir / "01_gyro")), lr=args.lr_Gyro, weight_decay=args.weight_decay_Gyro, epoch_count=recipe["gyro_updates"])
        ba = train_mse_acc(train, val, replace(config, outputdir=str(outputdir / "02_acc")), gyro_checkpoint=bg, lr=args.lr_Acc, weight_decay=args.weight_decay_Acc, epoch_count=recipe["acc_updates"])
        nll = train_nll_alternating(train, val, replace(config, outputdir=str(outputdir / "03_nll")), gyro_checkpoint=bg, acc_checkpoint=ba, bias_lr=args.lr_NLL_bias, covariance_lr=args.lr_NLL_covariance, bias_weight_decay=args.weight_decay_NLL, covariance_weight_decay=args.weight_decay_NLL_covariance, epoch_count=recipe["nll_updates"])
        manifest.update(status="trained", training_seconds=time.perf_counter()-started, checkpoints={name: {"path": str(Path(path).relative_to(outputdir)), "sha256": sha256(Path(path))} for name, path in (("gyro",bg),("acc",ba),("nll",nll))})
        write_json(manifest_path, manifest)
        if options.run_test:
            subprocess.run([sys.executable, str(REPO / "tools/evaluate_aqualux_release.py"), "--checkpoint", nll, "--data-dir", str(data_dir), "--cache-dir", str(outputdir / "cache"), "--outputdir", str(outputdir / "evaluation"), "--device", options.device], check=True)
            manifest["status"] = "evaluated"
            write_json(manifest_path, manifest)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(manifest_path, manifest)
        raise
    print(f"Completed: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
