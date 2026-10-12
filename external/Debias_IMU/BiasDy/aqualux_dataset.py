from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

import Interpolation
import lie_algebra as Lie
from dataset import BaseDataset, pdump, pload


def _data_root_for_module(module_path: Path) -> Path:
    """Use the checkout root in source, or the working directory after installation."""
    module_dir = module_path.resolve().parent
    if (module_dir.name, module_dir.parent.name, module_dir.parent.parent.name) == (
        "BiasDy", "Debias_IMU", "external"
    ):
        return module_dir.parents[2]
    return Path.cwd()


REPO_ROOT = _data_root_for_module(Path(__file__))
DEFAULT_DATA_DIR = os.environ.get(
    "AQUALUX_DATA_ROOT",
    str(REPO_ROOT / "datasets" / "AquaLux" / "0524_data"),
)
DEFAULT_GRAVITY_NORM = 9.7879
AQUALUX_CACHE_VERSION = 2
AQUALUX_QUAT_ORDER = "xyzw"


def sorted_aqualux_sequences(data_dir: str | os.PathLike, prefix: str) -> list[str]:
    seq_root = Path(data_dir) / "0524"
    return sorted(
        [p.name for p in seq_root.iterdir() if p.is_dir() and p.name.startswith(prefix)],
        key=lambda name: int(name.split("_", 1)[1]),
    )


def estimate_native_dt(data_dir: str | os.PathLike, sequences: list[str]) -> float:
    dt_parts: list[np.ndarray] = []
    for sequence in sequences:
        stamps, _ = read_imu_csv(_imu_path(data_dir, sequence))
        dt = np.diff(stamps)
        dt_parts.append(dt[dt > 0])
    if not dt_parts:
        raise ValueError("No sequences available for native dt estimation")
    return float(np.median(np.concatenate(dt_parts)))


def analyze_timing(data_dir: str | os.PathLike, sequences: list[str]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for sequence in sequences:
        imu_raw = np.genfromtxt(_imu_path(data_dir, sequence), delimiter=",", names=True, dtype=float)
        if imu_raw.ndim == 0:
            imu_raw = imu_raw.reshape(1)
        imu_stamps = np.asarray(imu_raw["timestamp"], dtype=float)
        gt_stamps, _, _ = read_tum(_gt_path(data_dir, sequence))
        imu_dt = np.diff(imu_stamps)
        gt_dt = np.diff(gt_stamps)
        overlap = min(float(imu_stamps[-1]), float(gt_stamps[-1])) - max(float(imu_stamps[0]), float(gt_stamps[0]))
        rows.append(
            {
                "sequence": sequence,
                "imu_samples": int(len(imu_stamps)),
                "imu_duplicate_dt": int(np.sum(imu_dt == 0)),
                "imu_time_backjump": int(np.sum(imu_dt < 0)),
                "imu_dt_min": float(np.min(imu_dt)),
                "imu_dt_median": float(np.median(imu_dt)),
                "imu_dt_max": float(np.max(imu_dt)),
                "gt_samples": int(len(gt_stamps)),
                "gt_duplicate_dt": int(np.sum(gt_dt == 0)),
                "gt_time_backjump": int(np.sum(gt_dt < 0)),
                "gt_dt_min": float(np.min(gt_dt)),
                "gt_dt_median": float(np.median(gt_dt)),
                "gt_dt_max": float(np.max(gt_dt)),
                "overlap_sec": float(overlap),
            }
        )
    return rows


def write_timing_csv(path: str | os.PathLike, rows: list[dict[str, float | int | str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sequence",
        "imu_samples",
        "imu_duplicate_dt",
        "imu_time_backjump",
        "imu_dt_min",
        "imu_dt_median",
        "imu_dt_max",
        "gt_samples",
        "gt_duplicate_dt",
        "gt_time_backjump",
        "gt_dt_min",
        "gt_dt_median",
        "gt_dt_max",
        "overlap_sec",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class AquaLuxDataset(BaseDataset):
    """BiasDy dataset adapter for AquaLux ATO GT and left IMU CSV files."""

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        data_cache_dir: str,
        train_seqs: list[str],
        val_seqs: list[str],
        test_seqs: list[str],
        mode: str,
        percent_for_val: float,
        batch_size: int,
        loss_window: int,
        dt: float,
        sg_window_bw: int,
        sg_order_bw: int,
        sg_window_ba: int,
        sg_order_ba: int,
        gravity_norm: float = DEFAULT_GRAVITY_NORM,
        gt_valid_max_dt: float = 0.02,
        velocity_smooth_sec: float = 0.12,
        velocity_smooth_polyorder: int = 3,
        recompute: bool = False,
    ):
        os.makedirs(data_cache_dir, exist_ok=True)
        super().__init__(
            dataset_name,
            data_cache_dir,
            train_seqs,
            val_seqs,
            test_seqs,
            mode,
            percent_for_val,
            batch_size,
            loss_window,
            dt,
        )
        self._sg_window_bw = sg_window_bw
        self._sg_order_bw = sg_order_bw
        self._sg_window_ba = sg_window_ba
        self._sg_order_ba = sg_order_ba
        self.gravity_norm = float(gravity_norm)
        self.gravity = torch.tensor([0.0, 0.0, -self.gravity_norm], dtype=torch.float32)
        self.gt_valid_max_dt = float(gt_valid_max_dt)
        self.velocity_smooth_sec = float(velocity_smooth_sec)
        self.velocity_smooth_polyorder = int(velocity_smooth_polyorder)
        self.read_data(data_dir, recompute)

    def read_data(self, data_dir, recompute=False):
        sequences = sorted(set(self._sequences_dict["train"] + self._sequences_dict["val"] + self._sequences_dict["test"]))
        if sequences and not recompute and all(self._cache_valid(sequence) for sequence in sequences):
            return

        print("Start AquaLux read_data at native IMU rate")
        for sequence in sequences:
            print("\nSequence name: " + sequence)
            self._convert_sequence(data_dir, sequence)

    def _cache_valid(self, sequence: str) -> bool:
        imu_cache = Path(self.data_cache_dir) / f"{sequence}.p"
        gt_cache = Path(self.data_cache_dir) / f"{sequence}_gt.p"
        if not imu_cache.exists() or not gt_cache.exists():
            return False
        try:
            imudict = pload(self.data_cache_dir, f"{sequence}.p")
            gtdict = pload(self.data_cache_dir, f"{sequence}_gt.p")
        except Exception:
            return False
        meta = imudict.get("metadata", {})
        gt_meta = gtdict.get("metadata", {})
        return (
            meta.get("dataset") == "AquaLux"
            and gt_meta.get("dataset") == "AquaLux"
            and int(meta.get("cache_version", -1)) == AQUALUX_CACHE_VERSION
            and int(gt_meta.get("cache_version", -1)) == AQUALUX_CACHE_VERSION
            and meta.get("quat_order") == AQUALUX_QUAT_ORDER
            and gt_meta.get("quat_order") == AQUALUX_QUAT_ORDER
            and math.isclose(float(meta.get("dt", -1.0)), float(self.dt), rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(float(gt_meta.get("dt", -1.0)), float(self.dt), rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(float(meta.get("gravity_norm", -1.0)), self.gravity_norm, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(float(gt_meta.get("gravity_norm", -1.0)), self.gravity_norm, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(float(meta.get("velocity_smooth_sec", -1.0)), self.velocity_smooth_sec, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(float(gt_meta.get("velocity_smooth_sec", -1.0)), self.velocity_smooth_sec, rel_tol=0.0, abs_tol=1e-9)
        )

    def _convert_sequence(self, data_dir: str | os.PathLike, sequence: str) -> None:
        imu_stamps, imu_raw = read_imu_csv(_imu_path(data_dir, sequence))
        gt_stamps, gt_positions_raw, gt_rotations_raw = read_tum(_gt_path(data_dir, sequence))

        start = max(float(imu_stamps[0]), float(gt_stamps[0]))
        end = min(float(imu_stamps[-1]), float(gt_stamps[-1]))
        query_abs = make_uniform_grid(start, end, self.dt)
        timestamp_start = float(query_abs[0])
        ts = query_abs - timestamp_start

        imu = interpolate_imu(imu_stamps, imu_raw, query_abs)
        p_gt, q_gt, gt_nearest_dt = interpolate_gt(gt_stamps, gt_positions_raw, gt_rotations_raw, query_abs)
        p_gt = p_gt - p_gt[0]
        v_gt = smooth_velocity(p_gt, self.dt, self.velocity_smooth_sec, self.velocity_smooth_polyorder)

        ts_t = torch.from_numpy(ts).double()
        imu_t = torch.from_numpy(imu).double()
        q_gt_t = torch.from_numpy(q_gt).double()
        q_gt_t = q_gt_t / q_gt_t.norm(dim=1, keepdim=True)
        rot_gt = Lie.quat_to_SO3(q_gt_t, order=AQUALUX_QUAT_ORDER).cpu()
        p_gt_t = torch.from_numpy(p_gt).double()
        v_gt_t = torch.from_numpy(v_gt).double()

        w_bias = imu_t[:-1, :3] - Lie.SO3log(rot_gt[:-1].transpose(-1, -2) @ rot_gt[1:]) / (
            ts_t[1:] - ts_t[:-1]
        ).unsqueeze(-1)
        w_bias = torch.cat((w_bias, w_bias[-1].unsqueeze(0)), dim=0)
        w_bias = torch.from_numpy(self.apply_sg_smoother(w_bias.cpu().numpy(), self._sg_window_bw, self._sg_order_bw))

        gravity = torch.tensor([0.0, 0.0, -self.gravity_norm], dtype=torch.double)
        a_bias = imu_t[:-1, 3:6] - (
            rot_gt[:-1].transpose(-1, -2)
            @ (
                (v_gt_t[1:] - v_gt_t[:-1]) / (ts_t[1:] - ts_t[:-1]).unsqueeze(-1)
                - gravity.unsqueeze(0)
            ).unsqueeze(-1)
        ).squeeze(-1)
        a_bias = torch.cat((a_bias, a_bias[-1].unsqueeze(0)), dim=0)
        a_bias = torch.from_numpy(self.apply_sg_smoother(a_bias.cpu().numpy(), self._sg_window_ba, self._sg_order_ba))

        coeff = Interpolation.get_cubic_hermite_coeffs(ts_t, imu_t)
        coeff = tuple(part.float() for part in coeff)

        gt_nearest_dt_t = torch.from_numpy(gt_nearest_dt).float()
        sample_mask_missing = gt_nearest_dt_t > self.gt_valid_max_dt
        mask_missing = build_window_mask(sample_mask_missing.cpu().numpy(), self._loss_window)
        metadata = {
            "dataset": "AquaLux",
            "cache_version": AQUALUX_CACHE_VERSION,
            "quat_order": AQUALUX_QUAT_ORDER,
            "sequence": sequence,
            "dt": float(self.dt),
            "rate_hz": float(1.0 / self.dt),
            "gravity_norm": self.gravity_norm,
            "gt_valid_max_dt": self.gt_valid_max_dt,
            "velocity_smooth_sec": self.velocity_smooth_sec,
            "source_imu_start": float(imu_stamps[0]),
            "source_imu_end": float(imu_stamps[-1]),
            "source_gt_start": float(gt_stamps[0]),
            "source_gt_end": float(gt_stamps[-1]),
        }

        pdump(
            {
                "t_imu": ts_t,
                "u_imu": imu_t.float(),
                "w_bias": w_bias.float(),
                "a_bias": a_bias.float(),
                "coeff": coeff,
                "t_start": timestamp_start,
                "gt_nearest_dt": gt_nearest_dt_t,
                "sample_mask_missing": sample_mask_missing.float(),
                "mask_missing": torch.from_numpy(mask_missing).float(),
                "metadata": metadata,
            },
            self.data_cache_dir,
            sequence + ".p",
        )
        pdump(
            {
                "t_gt": ts_t.float(),
                "q_gt": q_gt_t.float(),
                "R_gt": rot_gt.float(),
                "v_gt": v_gt_t.float(),
                "p_gt": p_gt_t.float(),
                "metadata": metadata,
            },
            self.data_cache_dir,
            sequence + "_gt.p",
        )


def _imu_path(data_dir: str | os.PathLike, sequence: str) -> Path:
    return Path(data_dir) / "0524" / sequence / "imu_diagnostics" / "imu_left.csv"


def _gt_path(data_dir: str | os.PathLike, sequence: str) -> Path:
    return Path(data_dir) / "0524" / sequence / "ATO" / "gt_imu_gravity_aligned.tum"


def read_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if rows.ndim == 0:
        rows = rows.reshape(1)
    required = ("timestamp", "wx", "wy", "wz", "ax", "ay", "az")
    if rows.dtype.names is None or any(name not in rows.dtype.names for name in required):
        raise ValueError(f"{path}: missing required IMU columns")
    stamps = np.asarray(rows["timestamp"], dtype=float)
    imu = np.column_stack([rows[name] for name in required[1:]]).astype(float)
    order = np.argsort(stamps)
    stamps = stamps[order]
    imu = imu[order]
    keep = np.r_[True, np.diff(stamps) > 0]
    return stamps[keep], imu[keep]


def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray, Rotation]:
    stamps: list[float] = []
    positions: list[list[float]] = []
    quats: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"{path}:{line_no}: expected TUM format with 8 fields")
            values = [float(part) for part in parts]
            stamps.append(values[0])
            positions.append(values[1:4])
            quats.append(values[4:8])
    stamps_arr = np.asarray(stamps, dtype=float)
    order = np.argsort(stamps_arr)
    stamps_arr = stamps_arr[order]
    if np.any(np.diff(stamps_arr) <= 0):
        raise ValueError(f"{path}: TUM timestamps must be strictly increasing")
    positions_arr = np.asarray(positions, dtype=float)[order]
    quats_arr = np.asarray(quats, dtype=float)[order]
    quats_arr = quats_arr / np.linalg.norm(quats_arr, axis=1, keepdims=True)
    return stamps_arr, positions_arr, Rotation.from_quat(quats_arr)


def make_uniform_grid(start: float, end: float, dt: float) -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be positive")
    count = int(math.floor((end - start) / dt)) + 1
    if count < 2:
        raise ValueError("No valid AquaLux IMU/GT overlap")
    return float(start) + np.arange(count, dtype=float) * float(dt)


def interpolate_imu(stamps: np.ndarray, imu: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(query, stamps, imu[:, axis]) for axis in range(6)])


def interpolate_gt(
    gt_stamps: np.ndarray,
    gt_positions: np.ndarray,
    gt_rotations: Rotation,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.column_stack([np.interp(query, gt_stamps, gt_positions[:, axis]) for axis in range(3)])
    rotations = Slerp(gt_stamps, gt_rotations)(query)
    nearest_idx = np.searchsorted(gt_stamps, query)
    nearest_idx = np.clip(nearest_idx, 1, len(gt_stamps) - 1)
    left = gt_stamps[nearest_idx - 1]
    right = gt_stamps[nearest_idx]
    nearest_dt = np.minimum(np.abs(query - left), np.abs(right - query))
    return positions, rotations.as_quat(), nearest_dt


def smooth_velocity(positions: np.ndarray, dt: float, smooth_sec: float, polyorder: int) -> np.ndarray:
    window = max(polyorder + 2, int(round(smooth_sec / dt)))
    if window % 2 == 0:
        window += 1
    if window >= len(positions):
        window = len(positions) - 1 if len(positions) % 2 == 0 else len(positions)
    if window <= polyorder or window < 3:
        return np.gradient(positions, dt, axis=0, edge_order=1)
    velocity = np.zeros_like(positions)
    for axis in range(3):
        velocity[:, axis] = savgol_filter(
            positions[:, axis],
            window,
            polyorder,
            deriv=1,
            delta=dt,
            mode="interp",
        )
    return velocity


def build_window_mask(sample_invalid: np.ndarray, loss_window: int) -> np.ndarray:
    invalid = np.asarray(sample_invalid, dtype=bool)
    mask = invalid.copy()
    if len(invalid) == 0 or loss_window <= 0:
        return mask
    csum = np.r_[0, np.cumsum(invalid.astype(np.int64))]
    last_start = max(0, len(invalid) - loss_window)
    for start in range(last_start):
        if csum[start + loss_window + 1] - csum[start] > 0:
            mask[start] = True
    return mask
