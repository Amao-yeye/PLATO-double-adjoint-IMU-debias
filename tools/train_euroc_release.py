#!/usr/bin/env python3
"""Train the public EuRoC recipe from scratch; evaluate only on explicit request.

Configuration and final-step selection are fixed before seeing test metrics.
All stages call the existing production training functions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from dataclasses import replace

REPO = Path(__file__).resolve().parents[1]
BIASDY = REPO / "external" / "Debias_IMU" / "BiasDy"
sys.path[:0] = [str(BIASDY), str(BIASDY.parent)]

import torch
from dataset import EUROCDataset
import mainEurocPLATO as euroc
from plato.training import train_mse_acc, train_mse_gyro, train_nll_alternating


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_parameters(recipe: dict, outputdir: Path, data_dir: Path, device: str):
    allowed = {"dataset", "gyro_updates", "acc_updates", "nll_updates", "checkpoint_selection", "parameters"}
    unknown = set(recipe) - allowed
    if unknown or recipe.get("dataset") != "euroc":
        raise ValueError(f"Expected an EuRoC recipe; unknown fields: {sorted(unknown)}")
    if recipe.get("checkpoint_selection") != "final":
        raise ValueError("This recipe selects predeclared final steps; checkpoint_selection must be 'final'")
    for name in ("gyro_updates", "acc_updates", "nll_updates"):
        if type(recipe.get(name)) is not int or recipe[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    parser = euroc.build_parser()
    actions = {action.dest: action for action in parser._actions}
    argv = []
    controlled = {"data_dir", "data_cache_dir", "outputdir", "device", "test_device", "resume_gyro", "resume_acc", "run_test", "run_compare", "plato_stage", "debug_smoke", "epoch_mse", "epoch_nll"}
    for name, value in recipe["parameters"].items():
        if name not in actions or name in controlled:
            raise ValueError(f"Unsupported recipe parameter: {name}")
        action = actions[name]
        option = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean")
            if value:
                argv.append(option)
        else:
            argv.append(option)
            argv.extend(str(v) for v in value) if isinstance(value, list) else argv.append(str(value))
    argv.extend(["--data_dir", str(data_dir), "--data_cache_dir", str(outputdir / "cache"), "--outputdir", str(outputdir), "--device", device])
    args = parser.parse_args(argv)
    args.loss_window = args.plato_window_steps
    if args.plato_supervision != "gt_se23":
        raise ValueError("The EuRoC release recipe currently supports GT supervision")
    if args.plato_sup_stride is not None or args.plato_sup_rate_hz is not None:
        raise ValueError("GT MSE must retain native-rate supervision")
    if args.nll_gt_sup_stride != 4:
        raise ValueError("The release GT NLL protocol requires stride 4")
    if set(args.train_seqs + args.val_seqs) & set(args.test_seqs):
        raise ValueError("Test sequences must be disjoint from training/validation sequences")
    return args, euroc.make_train_config(args)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/euroc_release.json")
    parser.add_argument("--data-dir", type=Path, default=REPO / "external/Debias_IMU/data/EUROC")
    parser.add_argument("--outputdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume-gyro", type=Path, help="Use an existing native-rate gyro MSE checkpoint and skip gyro training")
    parser.add_argument("--resume-acc", type=Path, help="Use an existing native-rate accelerometer MSE checkpoint and skip accelerometer training; requires --resume-gyro")
    parser.add_argument("--run-test", action="store_true", help="Evaluate the fixed final NLL checkpoint on all five held-out sequences")
    options = parser.parse_args()
    if options.threads < 1:
        parser.error("--threads must be positive")
    if bool(options.resume_gyro) != bool(options.resume_acc):
        parser.error("--resume-gyro and --resume-acc must be supplied together")
    for path in (options.resume_gyro, options.resume_acc):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    torch.set_num_threads(options.threads)
    recipe = json.loads(options.config.read_text(encoding="utf-8"))
    outputdir = options.outputdir.resolve()
    args, config = resolve_parameters(recipe, outputdir, options.data_dir.resolve(), options.device)
    if outputdir.exists() and any(outputdir.iterdir()):
        raise FileExistsError(f"Output directory must be empty to protect existing experiments: {outputdir}")
    outputdir.mkdir(parents=True, exist_ok=True)
    source_files = sorted(path for base in (BIASDY, REPO / "tools") for path in base.rglob("*.py"))
    manifest = {
        "status": "running", "recipe": recipe, "resolved_arguments": vars(args),
        "checkpoint_selection": "predeclared final optimizer step; no test-driven selection",
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "device": options.device,
        "threads": options.threads, "seed_per_stage": 3407,
        "mse_initialization": "provided native-rate checkpoints" if options.resume_gyro else "from scratch",
        "executed_optimizer_updates": {"gyro": 0 if options.resume_gyro else recipe["gyro_updates"], "acc": 0 if options.resume_acc else recipe["acc_updates"], "nll": recipe["nll_updates"]},
        "source_sha256": {str(path.relative_to(REPO)): sha256(path) for path in source_files},
        "recipe_sha256": sha256(options.config),
        "data_sha256": {f"{seq}/{kind}": sha256(options.data_dir / seq / "mav0" / kind / "data.csv") for seq in dict.fromkeys(args.train_seqs + args.val_seqs + args.test_seqs) for kind in ("imu0", "state_groundtruth_estimate0")},
    }
    if options.device.startswith("cuda"):
        manifest["gpu"] = torch.cuda.get_device_name(options.device)
    manifest_path = outputdir / "run_manifest.json"
    write_json(manifest_path, manifest)
    started = time.perf_counter()
    try:
        parameters = euroc.euroc_dataset_parameters(args)
        train = EUROCDataset(**parameters, mode="train")
        val = EUROCDataset(**parameters, mode="val")
        if options.resume_gyro:
            bg, ba = str(options.resume_gyro.resolve()), str(options.resume_acc.resolve())
            for path in (bg, ba):
                metadata = torch.load(path, map_location="cpu", weights_only=False).get("metadata", {})
                if metadata.get("supervision") != "gt_se23" or metadata.get("sup_stride") is not None or metadata.get("mean_discretization") != config.mean_discretization:
                    raise ValueError("MSE continuation requires compatible native-rate GT checkpoints")
                if set(args.test_seqs) & set([*metadata.get("train_seqs", ()), *metadata.get("val_seqs", ())]):
                    raise ValueError("MSE checkpoint overlaps with held-out test sequences")
        else:
            bg = train_mse_gyro(train, val, replace(config, outputdir=str(outputdir / "01_gyro")), lr=args.lr_Gyro, weight_decay=args.weight_decay_Gyro, epoch_count=recipe["gyro_updates"])
            ba = train_mse_acc(train, val, replace(config, outputdir=str(outputdir / "02_acc")), gyro_checkpoint=bg, lr=args.lr_Acc, weight_decay=args.weight_decay_Acc, epoch_count=recipe["acc_updates"])
        nll = train_nll_alternating(train, val, replace(config, outputdir=str(outputdir / "03_nll")), gyro_checkpoint=bg, acc_checkpoint=ba, bias_lr=args.lr_NLL, covariance_lr=args.lr_covariance, bias_weight_decay=args.weight_decay_NLL, covariance_weight_decay=0.0, epoch_count=recipe["nll_updates"], block_epochs=args.nll_block_epochs, start_with=args.nll_start_with)
        manifest["checkpoints"] = {name: {"path": str(Path(path).relative_to(outputdir)) if Path(path).is_relative_to(outputdir) else str(Path(path)), "sha256": sha256(Path(path))} for name, path in (("gyro", bg), ("acc", ba), ("nll", nll))}
        manifest["training_seconds"] = time.perf_counter() - started
        manifest["status"] = "trained"
        write_json(manifest_path, manifest)
        if options.run_test:
            # Evaluation is a separate process so its result has independent provenance.
            subprocess.run([sys.executable, str(REPO / "tools/evaluate_euroc_release.py"), "--checkpoint", nll, "--data-dir", args.data_dir, "--cache-dir", args.data_cache_dir, "--outputdir", str(outputdir / "evaluation"), "--device", options.device], check=True)
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
