#!/usr/bin/env python3
"""Export a portable inference checkpoint without optimizer state or private paths."""
from __future__ import annotations

import argparse
from pathlib import Path

from train_euroc_release import sha256
import torch


def export(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    for name in ("func_bw_model", "func_ba_model", "metadata"):
        if name not in checkpoint:
            raise ValueError(f"Missing inference field: {name}")
    fields = ("func_bw_model", "func_ba_model", "noise_params", "learned_imu_covariance", "learned_ba_rate_limit", "epoch", "valloss", "alternating_protocol")
    portable = {name: checkpoint[name] for name in fields if name in checkpoint}
    portable["metadata"] = {name: value for name, value in checkpoint["metadata"].items() if name not in ("outputdir", "device")}
    portable["source_checkpoint_sha256"] = sha256(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(portable, output)
    return {"sha256": sha256(output), "source_checkpoint_sha256": portable["source_checkpoint_sha256"], "epoch": checkpoint.get("epoch")}


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.checkpoint, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
