#!/usr/bin/env python3
"""Validate the AquaLux files required by the formal PLATO experiments."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("AQUALUX_DATA_ROOT", REPO_ROOT / "datasets" / "AquaLux" / "0524_data")
)
DEFAULT_TRAIN = ("Train_000", "Train_002", "Train_008", "Train_010", "Train_012", "Train_014", "Train_016", "Train_018")
DEFAULT_TEST = tuple(f"Test_{index:03d}" for index in range(1, 20, 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_TRAIN + DEFAULT_TEST))
    parser.add_argument("--require-vo", action="store_true")
    parser.add_argument("--require-bags", action="store_true")
    return parser.parse_args()


def read_csv_timestamps(path: Path) -> tuple[int, float, float]:
    timestamps: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "timestamp" not in reader.fieldnames:
            raise ValueError("missing 'timestamp' CSV column")
        for row in reader:
            timestamps.append(float(row["timestamp"]))
    if len(timestamps) < 2:
        raise ValueError("fewer than two IMU samples")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("IMU timestamps are not strictly increasing")
    return len(timestamps), timestamps[0], timestamps[-1]


def read_tum_timestamps(path: Path, *, allow_duplicates: bool = False) -> tuple[int, float, float]:
    timestamps: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 8:
                raise ValueError("TUM row has fewer than eight fields")
            timestamps.append(float(fields[0]))
    if len(timestamps) < 2:
        raise ValueError("fewer than two trajectory samples")
    if allow_duplicates:
        if any(right < left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("trajectory timestamps contain a backward jump")
        if timestamps[-1] <= timestamps[0]:
            raise ValueError("trajectory has fewer than two unique timestamps")
    elif any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("trajectory timestamps are not strictly increasing")
    return len(timestamps), timestamps[0], timestamps[-1]


def main() -> int:
    args = parse_args()
    root = args.data_root.resolve()
    failures: list[str] = []
    checked = 0

    if not (root / "0524").is_dir():
        raise SystemExit(f"AquaLux sequence root not found: {root / '0524'}")

    for sequence in args.sequences:
        sequence_root = root / "0524" / sequence
        required = {
            "imu": sequence_root / "imu_diagnostics" / "imu_left.csv",
            "gt": sequence_root / "ATO" / "gt_imu_gravity_aligned.tum",
        }
        if args.require_vo:
            required["vo"] = sequence_root / "ATO" / "vo_imu_gravity_aligned.tum"
        if args.require_bags:
            required["bag"] = root / f"{sequence}.bag"

        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            failures.append(f"{sequence}: missing {', '.join(missing)}")
            continue
        try:
            imu_count, imu_start, imu_end = read_csv_timestamps(required["imu"])
            gt_count, gt_start, gt_end = read_tum_timestamps(required["gt"])
            if min(imu_end, gt_end) <= max(imu_start, gt_start):
                raise ValueError("IMU and GT time ranges do not overlap")
            if "vo" in required:
                # AquaLux VO exports can contain repeated image timestamps; the
                # training loader accepts them as long as time never jumps back.
                read_tum_timestamps(required["vo"], allow_duplicates=True)
        except (OSError, ValueError) as error:
            failures.append(f"{sequence}: {error}")
            continue
        checked += 1
        print(f"OK {sequence}: imu={imu_count}, gt={gt_count}")

    if failures:
        print("\nAquaLux validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"AquaLux validation passed: {checked} sequences under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
