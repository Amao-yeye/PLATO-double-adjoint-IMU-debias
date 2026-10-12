#!/usr/bin/env python3
"""Compare a complete per-sequence CSV with the supplied manuscript's results.

This numerical gate does not certify data independence, initialization, alignment,
or training provenance. Those conditions must be recorded by the evaluator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REFERENCE_PATH = Path(__file__).resolve().parents[1] / "configs/paper_metrics.json"


def compare(rows: list[dict], reference: dict, duration_policy: str = "paper") -> dict:
    if duration_policy not in ("paper", "input_overlap"):
        raise ValueError("duration_policy must be paper or input_overlap")
    expected = reference["sequences"]
    names = [row["sequence"] for row in rows]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError("Metrics must contain each reference test sequence exactly once, without extra rows")
    metrics = reference["metrics"]
    values = {}
    duration_differences = {}
    for row in rows:
        for metric in metrics:
            value = float(row[metric])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {metric} for {row['sequence']}: {value}")
        duration = expected[row["sequence"]].get("duration_seconds")
        if duration is not None:
            observed = float(row["duration_seconds"])
            if not math.isfinite(observed) or observed <= 0:
                raise ValueError(f"Invalid duration for {row['sequence']}")
            if abs(observed - duration) > 0.01:
                duration_differences[row["sequence"]] = {"measured": observed, "paper": duration}
                if duration_policy == "paper":
                    raise ValueError(f"Duration mismatch for {row['sequence']}: got {observed}, paper {duration}")
            if duration_policy == "input_overlap":
                expected_duration = float(row["input_overlap_duration_seconds"])
                expected_count = int(row["input_overlap_sample_count"])
                if not math.isfinite(expected_duration) or abs(observed - expected_duration) > 1e-4 or expected_count < 2 or int(row["sample_count"]) != expected_count:
                    raise ValueError(f"Rollout does not cover the complete raw IMU/GT overlap for {row['sequence']}")
    for metric in metrics:
        actual = math.fsum(float(row[metric]) for row in rows) / len(rows)
        baseline = math.fsum(value[metric] for value in expected.values()) / len(expected)
        values[metric] = {"measured": actual, "paper": baseline, "improvement_percent": 100 * (baseline - actual) / baseline, "better": actual < baseline}
    return {"all_metrics_better": all(value["better"] for value in values.values()), "aggregation": "unweighted arithmetic mean of per-sequence RMSEs", "sequence_count": len(rows), "metrics": values, "duration_policy": duration_policy, "paper_duration_differences": duration_differences}


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--reference", choices=["euroc_gt_preintegration", "euroc_gt_openvins", "aqualux_vo_preintegration", "aqualux_gt_hlcbd_preintegration", "aqualux_vo_openvins"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration-policy", choices=["paper", "input_overlap"], default="paper", help="input_overlap requires raw-overlap coverage fields and reports paper-duration differences explicitly")
    args = parser.parse_args()
    with args.metrics.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    references = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    result = compare(rows, references[args.reference], args.duration_policy)
    result["reference"] = args.reference
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["all_metrics_better"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
