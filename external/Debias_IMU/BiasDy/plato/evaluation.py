from __future__ import annotations

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

from dataset import BaseDataset, pload
from evalutation import AOE, ATE


def summarize_comparison_metrics(dataset_test: BaseDataset, outputdir: str) -> list[dict[str, float | str]]:
    """Summarize raw/proposed/Axb metrics from exported trajectory files."""
    rows = []
    for seq in dataset_test.sequences:
        proposed_path = os.path.join(outputdir, "Sequnces-results", seq, "FullTra_result.p")
        raw_path = os.path.join(outputdir, "raw_int", seq + ".p")
        if not os.path.exists(proposed_path):
            raise FileNotFoundError(f"Missing proposed result: {proposed_path}")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Missing raw integration result: {raw_path}")

        proposed = pload(proposed_path)
        raw = pload(raw_path)
        X_gt = proposed["X_gt"]
        R_pred = proposed["R_pred"]
        v_pred = proposed["sol_pred"][:, 7:10]
        p_pred = proposed["sol_pred"][:, 10:13]
        X_raw = raw["X_raw"]

        row = {
            "sequence": seq,
            "raw_aoe_deg": _to_float(AOE(X_gt[:, :3, :3], X_raw[:, :3, :3])),
            "raw_vel_rmse": _to_float(ATE(X_gt[:, :3, 3], X_raw[:, :3, 3])),
            "raw_ape_m": _to_float(ATE(X_gt[:, :3, 4], X_raw[:, :3, 4])),
            "plato_aoe_deg": _to_float(AOE(X_gt[:, :3, :3], R_pred)),
            "plato_vel_rmse": _to_float(ATE(X_gt[:, :3, 3], v_pred)),
            "plato_ape_m": _to_float(ATE(X_gt[:, :3, 4], p_pred)),
        }

        axb_path = os.path.join(outputdir, "Axb_results", seq, "Axb_results.p")
        if os.path.exists(axb_path):
            axb = pload(axb_path)
            X_Axb = axb["X_Axb"]
            row.update({
                "axb_aoe_deg": _to_float(AOE(X_gt[:, :3, :3], X_Axb[:, :3, :3])),
                "axb_vel_rmse": _to_float(ATE(X_gt[:, :3, 3], X_Axb[:, :3, 3])),
                "axb_ape_m": _to_float(ATE(X_gt[:, :3, 4], X_Axb[:, :3, 4])),
            })
        rows.append(row)
    return rows


def write_metrics_csv(rows: list[dict[str, float | str]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = _fieldnames(rows)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_metrics_table(rows: list[dict[str, float | str]]) -> None:
    for row in rows:
        parts = [
            f"Seq {row['sequence']:16}",
            f"raw AOE {row['raw_aoe_deg']:.3f} deg",
            f"raw APE {row['raw_ape_m']:.3f} m",
            f"PLATO AOE {row['plato_aoe_deg']:.3f} deg",
            f"PLATO APE {row['plato_ape_m']:.3f} m",
        ]
        if "axb_aoe_deg" in row:
            parts.extend([
                f"Axb AOE {row['axb_aoe_deg']:.3f} deg",
                f"Axb APE {row['axb_ape_m']:.3f} m",
            ])
        print(" | ".join(parts))


def _fieldnames(rows: list[dict[str, float | str]]) -> list[str]:
    fields = [
        "sequence",
        "raw_aoe_deg",
        "raw_vel_rmse",
        "raw_ape_m",
        "plato_aoe_deg",
        "plato_vel_rmse",
        "plato_ape_m",
    ]
    if any("axb_aoe_deg" in row for row in rows):
        fields.extend(["axb_aoe_deg", "axb_vel_rmse", "axb_ape_m"])
    return fields


def _to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)
