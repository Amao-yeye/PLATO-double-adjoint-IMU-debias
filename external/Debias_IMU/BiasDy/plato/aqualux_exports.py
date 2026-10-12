"""Trajectory and metric exports shared by the AquaLux PLATO entry point."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import lie_algebra as Lie
from aqualux_dataset import AquaLuxDataset
from dataset import pload


def export_preintegration_results(dataset_test: AquaLuxDataset, outputdir: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for index, sequence in enumerate(dataset_test.sequences):
        result_path = outputdir / "Sequnces-results" / sequence / "FullTra_result.p"
        result = pload(result_path)
        seq_dir = outputdir / sequence
        seq_dir.mkdir(parents=True, exist_ok=True)

        t = to_numpy(result["t_gt"]).astype(np.float64)
        timestamp_start = float(result.get("timestamp_start", dataset_test.get_start_timestamp(index)))
        stamps = np.float64(timestamp_start) + t
        x_gt = result["X_gt"].detach().cpu()
        r_pred = result["R_pred"].detach().cpu()
        sol_pred = result["sol_pred"].detach().cpu()
        p_gt = x_gt[:, :3, 4]
        r_gt = x_gt[:, :3, :3]
        p_pred = sol_pred[:, 10:13]
        net_imu = result["net_imu"].detach().cpu()

        imudict = dataset_test.load_imu(index)
        gt_nearest_dt = to_numpy(imudict.get("gt_nearest_dt", torch.zeros_like(result["t_gt"])))
        valid = gt_nearest_dt <= dataset_test.gt_valid_max_dt
        n = min(len(valid), len(stamps))
        valid = valid[:n]
        stamps = stamps[:n]
        gt_nearest_dt = gt_nearest_dt[:n]
        p_gt = p_gt[:n]
        r_gt = r_gt[:n]
        p_pred = p_pred[:n]
        r_pred = r_pred[:n]
        net_imu = net_imu[:n]

        pos_err = torch.linalg.norm(p_pred - p_gt, dim=1).numpy()
        rot_err = (
            torch.linalg.norm(Lie.SO3log(r_gt @ r_pred.transpose(-1, -2)), dim=1) * 180.0 / torch.pi
        ).numpy()

        gt_path = seq_dir / "gt_resampled.tum"
        pred_path = seq_dir / "bias_dynamic_preintegrated.tum"
        errors_path = seq_dir / "errors.csv"
        imu_path = seq_dir / "corrected_imu.csv"
        write_tum(gt_path, stamps, p_gt.numpy(), r_gt.numpy())
        write_tum(pred_path, stamps, p_pred.numpy(), r_pred.numpy())
        write_corrected_imu_csv(imu_path, stamps, net_imu.numpy())
        write_errors_csv(errors_path, stamps, valid, gt_nearest_dt, pos_err, rot_err)

        if int(np.sum(valid)) < 10:
            raise RuntimeError(f"{sequence}: too few valid GT metric samples")
        rows.append(
            {
                "sequence": sequence,
                "samples": int(n),
                "valid_metric_samples": int(np.sum(valid)),
                "duration_sec": float(stamps[-1] - stamps[0]),
                "gt_interp_max_dt_sec": float(np.max(gt_nearest_dt)),
                "gt_valid_max_dt_sec": float(dataset_test.gt_valid_max_dt),
                "position_rmse_m": rmse(pos_err[valid]),
                "rotation_rmse_deg": rmse(rot_err[valid]),
                "position_mean_m": float(np.mean(pos_err[valid])),
                "rotation_mean_deg": float(np.mean(rot_err[valid])),
                "position_max_m": float(np.max(pos_err[valid])),
                "rotation_max_deg": float(np.max(rot_err[valid])),
                "gt_resampled": str(gt_path),
                "bias_dynamic_preintegrated": str(pred_path),
                "corrected_imu": str(imu_path),
                "errors_csv": str(errors_path),
            }
        )
    return rows


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def rmse(values: np.ndarray) -> float:
    if len(values) == 0:
        return math.nan
    return float(np.sqrt(np.mean(values * values)))


def write_tum(path: Path, stamps: np.ndarray, positions: np.ndarray, rotations: np.ndarray) -> None:
    quats = Rotation.from_matrix(rotations).as_quat()
    with path.open("w", encoding="utf-8") as f:
        for stamp, pos, quat in zip(stamps, positions, quats):
            f.write(
                f"{stamp:.9f} "
                f"{pos[0]:.12g} {pos[1]:.12g} {pos[2]:.12g} "
                f"{quat[0]:.12g} {quat[1]:.12g} {quat[2]:.12g} {quat[3]:.12g}\n"
            )


def write_corrected_imu_csv(path: Path, stamps: np.ndarray, imu: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "wx", "wy", "wz", "ax", "ay", "az"])
        for stamp, row in zip(stamps, imu):
            writer.writerow([f"{stamp:.9f}"] + [f"{value:.12g}" for value in row])


def write_errors_csv(
    path: Path,
    stamps: np.ndarray,
    valid: np.ndarray,
    gt_nearest_dt: np.ndarray,
    pos_err: np.ndarray,
    rot_err: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "valid_metric_sample", "gt_nearest_dt_sec", "position_error_m", "rotation_error_deg"])
        for row in zip(stamps, valid, gt_nearest_dt, pos_err, rot_err):
            writer.writerow([f"{row[0]:.9f}", int(row[1]), f"{row[2]:.12g}", f"{row[3]:.12g}", f"{row[4]:.12g}"])


def write_metrics_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fields = [
        "sequence",
        "samples",
        "valid_metric_samples",
        "duration_sec",
        "gt_interp_max_dt_sec",
        "gt_valid_max_dt_sec",
        "position_rmse_m",
        "rotation_rmse_deg",
        "position_mean_m",
        "rotation_mean_deg",
        "position_max_m",
        "rotation_max_deg",
        "gt_resampled",
        "bias_dynamic_preintegrated",
        "corrected_imu",
        "errors_csv",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, float | int | str]]) -> dict[str, float | int]:
    pos_parts: list[np.ndarray] = []
    rot_parts: list[np.ndarray] = []
    for row in rows:
        data = np.genfromtxt(row["errors_csv"], delimiter=",", names=True, dtype=float)
        if data.ndim == 0:
            data = data.reshape(1)
        valid = data["valid_metric_sample"] > 0.5
        pos_parts.append(np.asarray(data["position_error_m"][valid], dtype=float))
        rot_parts.append(np.asarray(data["rotation_error_deg"][valid], dtype=float))
    pos = np.concatenate(pos_parts)
    rot = np.concatenate(rot_parts)
    return {
        "valid_metric_samples": int(len(pos)),
        "position_rmse_m": rmse(pos),
        "rotation_rmse_deg": rmse(rot),
        "position_mean_m": float(np.mean(pos)),
        "rotation_mean_deg": float(np.mean(rot)),
        "position_max_m": float(np.max(pos)),
        "rotation_max_deg": float(np.max(rot)),
    }
