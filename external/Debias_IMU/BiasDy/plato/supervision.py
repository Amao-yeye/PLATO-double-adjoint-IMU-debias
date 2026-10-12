from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

BIASDY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BIASDY_DIR not in sys.path:
    sys.path.insert(0, BIASDY_DIR)

try:
    import Interpolation
    import lie_algebra as Lie
except ImportError:  # Support `python -m BiasDy.plato...` from Debias_IMU root.
    from BiasDy import Interpolation
    from BiasDy import lie_algebra as Lie


@dataclass
class PoseTrajectory:
    t: torch.Tensor
    R: torch.Tensor
    p: torch.Tensor
    v: torch.Tensor | None = None


@dataclass
class PlatoSupervisionBatch:
    mode: str
    t_state: torch.Tensor
    X_state: torch.Tensor
    bwa_state: torch.Tensor
    t_odeint: torch.Tensor
    coeff: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    spline_time: torch.Tensor
    starts: torch.Tensor
    supervised_indices: torch.Tensor
    t_ref: torch.Tensor
    X_ref: torch.Tensor | None = None
    R_ref: torch.Tensor | None = None
    p_ref: torch.Tensor | None = None


def get_gt_se23_supervision_batch(
    dataset,
    sequence_index: int,
    device: torch.device | str = "cpu",
    window_steps: int | None = None,
    batch_size: int | None = None,
    mask_flag: bool = False,
) -> PlatoSupervisionBatch:
    """Sample a dense EuRoC GT SE_2(3) supervision window."""
    window_steps = int(window_steps or dataset._loss_window)
    starts = sample_start_indices(
        dataset,
        sequence_index,
        window_steps=window_steps,
        batch_size=batch_size,
        mask_flag=mask_flag,
    )
    batch = gather_state_window(dataset, sequence_index, starts, window_steps, device=device)
    supervised_indices = torch.arange(window_steps, device=device, dtype=torch.long)
    return PlatoSupervisionBatch(
        mode="gt_se23",
        t_state=batch["t_state"],
        X_state=batch["X_state"],
        bwa_state=batch["bwa_state"],
        t_odeint=batch["t_odeint"],
        coeff=batch["coeff"],
        spline_time=batch["spline_time"],
        starts=starts.to(device),
        supervised_indices=supervised_indices,
        t_ref=batch["t_state"],
        X_ref=batch["X_state"],
        R_ref=batch["X_state"][..., :3, :3],
        p_ref=batch["X_state"][..., :3, 4],
    )


def get_vo_rate_gt_se23_supervision_batch(
    dataset,
    sequence_index: int,
    vo_traj: PoseTrajectory,
    device: torch.device | str = "cpu",
    window_steps: int | None = None,
    sup_stride: int | None = None,
    sup_rate_hz: float | None = None,
    batch_size: int | None = None,
    mask_flag: bool = False,
) -> PlatoSupervisionBatch:
    """Sample sparse GT on the same native-IMU grid used by VO supervision.

    ``window_steps`` counts supervised states (for example, W=64), not native
    IMU states.  VO contributes timing only: its median timestamp spacing (or
    an explicit stride/rate override) determines the native-IMU stride and its
    time support filters valid windows.  Every target state and the rollout
    initializer come from GT, while all intervening native-rate IMU states are
    retained in ``t_odeint``.
    """
    gt_window_samples = int(window_steps or dataset._loss_window)
    resolved_stride = resolve_vo_supervision_stride(
        dataset.dt,
        vo_traj,
        sup_stride=sup_stride,
        sup_rate_hz=sup_rate_hz,
    )
    imu_window_steps, supervised_indices = build_vo_supervised_indices(
        gt_window_samples,
        resolved_stride,
    )
    starts = sample_start_indices(
        dataset,
        sequence_index,
        window_steps=imu_window_steps,
        batch_size=batch_size,
        mask_flag=mask_flag,
        supervised_indices=supervised_indices,
        vo_traj=vo_traj,
    )
    batch = gather_state_window(
        dataset,
        sequence_index,
        starts,
        imu_window_steps,
        device=device,
    )
    supervised_indices = supervised_indices.to(device)
    X_ref = batch["X_state"][supervised_indices]
    t_ref = batch["t_state"][supervised_indices]
    return PlatoSupervisionBatch(
        mode="vo_rate_gt_se23",
        t_state=batch["t_state"],
        X_state=batch["X_state"],
        bwa_state=batch["bwa_state"],
        t_odeint=batch["t_odeint"],
        coeff=batch["coeff"],
        spline_time=batch["spline_time"],
        starts=starts.to(device),
        supervised_indices=supervised_indices,
        t_ref=t_ref,
        X_ref=X_ref,
        R_ref=X_ref[..., :3, :3],
        p_ref=X_ref[..., :3, 4],
    )


def get_vo_pose_supervision_batch(
    dataset,
    sequence_index: int,
    vo_traj: PoseTrajectory,
    device: torch.device | str = "cpu",
    window_steps: int | None = None,
    sup_stride: int | None = None,
    sup_rate_hz: float | None = None,
    batch_size: int | None = None,
    mask_flag: bool = False,
    include_window_end: bool = True,
    vo_init_mode: str = "gt",
) -> PlatoSupervisionBatch:
    """Sample an IMU window containing window_steps VO-rate pose samples."""
    vo_window_samples = int(window_steps or dataset._loss_window)
    sup_stride = resolve_vo_supervision_stride(
        dataset.dt, vo_traj, sup_stride=sup_stride, sup_rate_hz=sup_rate_hz
    )
    imu_window_steps, supervised_indices = build_vo_supervised_indices(
        vo_window_samples, sup_stride
    )
    starts = sample_start_indices(
        dataset,
        sequence_index,
        window_steps=imu_window_steps,
        batch_size=batch_size,
        mask_flag=mask_flag,
        supervised_indices=supervised_indices,
        vo_traj=vo_traj,
    )
    vo_traj = move_pose_trajectory(vo_traj, device=device, dtype=torch.float32)
    if vo_init_mode == "gt":
        batch = gather_state_window(dataset, sequence_index, starts, imu_window_steps, device=device)
        vo_traj = move_pose_trajectory(vo_traj, device=device, dtype=batch["X_state"].dtype)
    elif vo_init_mode == "vo_zero_bias":
        batch = gather_vo_state_window(
            dataset, sequence_index, starts, imu_window_steps, vo_traj, device=device
        )
    else:
        raise ValueError("vo_init_mode must be gt or vo_zero_bias")
    supervised_indices = supervised_indices.to(device)
    t_ref = batch["t_state"][supervised_indices]

    R_ref, p_ref = interpolate_pose(t_ref, vo_traj)
    return PlatoSupervisionBatch(
        mode="vo_pose",
        t_state=batch["t_state"],
        X_state=batch["X_state"],
        bwa_state=batch["bwa_state"],
        t_odeint=batch["t_odeint"],
        coeff=batch["coeff"],
        spline_time=batch["spline_time"],
        starts=starts.to(device),
        supervised_indices=supervised_indices,
        t_ref=t_ref,
        R_ref=R_ref,
        p_ref=p_ref,
    )


def get_vo_se23_supervision_batch(
    dataset,
    sequence_index: int,
    vo_traj: PoseTrajectory,
    device: torch.device | str = "cpu",
    window_steps: int | None = None,
    sup_stride: int | None = None,
    sup_rate_hz: float | None = None,
    batch_size: int | None = None,
    mask_flag: bool = False,
    include_window_end: bool = True,
    vo_init_mode: str = "gt",
) -> PlatoSupervisionBatch:
    """Sample an IMU window containing window_steps VO-rate SE_2(3) samples."""
    vo_window_samples = int(window_steps or dataset._loss_window)
    sup_stride = resolve_vo_supervision_stride(
        dataset.dt, vo_traj, sup_stride=sup_stride, sup_rate_hz=sup_rate_hz
    )
    imu_window_steps, supervised_indices = build_vo_supervised_indices(
        vo_window_samples, sup_stride
    )
    starts = sample_start_indices(
        dataset,
        sequence_index,
        window_steps=imu_window_steps,
        batch_size=batch_size,
        mask_flag=mask_flag,
        supervised_indices=supervised_indices,
        vo_traj=vo_traj,
    )
    vo_traj = move_pose_trajectory(vo_traj, device=device, dtype=torch.float32)
    if vo_init_mode == "gt":
        batch = gather_state_window(dataset, sequence_index, starts, imu_window_steps, device=device)
        vo_traj = move_pose_trajectory(vo_traj, device=device, dtype=batch["X_state"].dtype)
    elif vo_init_mode == "vo_zero_bias":
        batch = gather_vo_state_window(
            dataset, sequence_index, starts, imu_window_steps, vo_traj, device=device
        )
    else:
        raise ValueError("vo_init_mode must be gt or vo_zero_bias")
    supervised_indices = supervised_indices.to(device)
    t_ref = batch["t_state"][supervised_indices]

    R_ref, p_ref = interpolate_pose(t_ref, vo_traj)
    v_ref = interpolate_velocity(t_ref, vo_traj)
    X_ref = Lie.SEn3fromSO3Rn(R_ref, torch.cat((v_ref, p_ref), dim=-1))
    return PlatoSupervisionBatch(
        mode="vo_se23",
        t_state=batch["t_state"],
        X_state=batch["X_state"],
        bwa_state=batch["bwa_state"],
        t_odeint=batch["t_odeint"],
        coeff=batch["coeff"],
        spline_time=batch["spline_time"],
        starts=starts.to(device),
        supervised_indices=supervised_indices,
        t_ref=t_ref,
        X_ref=X_ref,
        R_ref=R_ref,
        p_ref=p_ref,
    )


def gather_state_window(
    dataset,
    sequence_index: int,
    starts: torch.Tensor,
    window_steps: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Gather dense GT state, pseudo-bias initializers, and spline data for ODE integration."""
    imudict = dataset.load_imu(sequence_index)
    gtdict = dataset.load_gt(sequence_index)
    starts = starts.to(dtype=torch.long, device=torch.device("cpu"))
    offsets = torch.arange(window_steps, dtype=torch.long)
    indices = starts.unsqueeze(0) + offsets.unsqueeze(1)

    R_gt = gtdict["R_gt"][indices]
    v_gt = gtdict["v_gt"][indices]
    p_gt = gtdict["p_gt"][indices]
    X_state = Lie.SEn3fromSO3Rn(R_gt, torch.cat((v_gt, p_gt), dim=-1))
    t_state = gtdict["t_gt"][indices]
    bw_state = imudict["w_bias"][indices]
    ba_state = imudict["a_bias"][indices]
    bwa_state = torch.cat((bw_state, ba_state), dim=-1)

    t_odeint = torch.arange(window_steps, dtype=t_state.dtype) * float(dataset.dt)
    coeff = tuple(c.to(device) for c in imudict["coeff"])
    return {
        "t_state": t_state.to(device),
        "X_state": X_state.to(device),
        "bwa_state": bwa_state.to(device),
        "t_odeint": t_odeint.to(device),
        "coeff": coeff,
        "spline_time": imudict["t_imu"].to(device),
    }


def gather_vo_state_window(
    dataset,
    sequence_index: int,
    starts: torch.Tensor,
    window_steps: int,
    vo_traj: PoseTrajectory,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Gather IMU timing/spline data and initialize the ODE state from VO only."""
    imudict = dataset.load_imu(sequence_index)
    starts = starts.to(dtype=torch.long, device=torch.device("cpu"))
    offsets = torch.arange(window_steps, dtype=torch.long)
    indices = starts.unsqueeze(0) + offsets.unsqueeze(1)

    t_state = imudict["t_imu"][indices].to(device=device, dtype=vo_traj.t.dtype)
    vo_traj = move_pose_trajectory(vo_traj, device=device, dtype=t_state.dtype)
    R_state, p_state = interpolate_pose(t_state, vo_traj)
    v_state = interpolate_velocity(t_state, vo_traj)
    X_state = Lie.SEn3fromSO3Rn(R_state, torch.cat((v_state, p_state), dim=-1))
    bwa_state = torch.zeros(*t_state.shape, 6, device=t_state.device, dtype=t_state.dtype)

    t_odeint = torch.arange(window_steps, dtype=t_state.dtype, device=t_state.device) * float(dataset.dt)
    coeff = tuple(c.to(device) for c in imudict["coeff"])
    return {
        "t_state": t_state,
        "X_state": X_state,
        "bwa_state": bwa_state,
        "t_odeint": t_odeint,
        "coeff": coeff,
        "spline_time": imudict["t_imu"].to(device=device, dtype=vo_traj.t.dtype),
    }


def sample_start_indices(
    dataset,
    sequence_index: int,
    window_steps: int,
    batch_size: int | None = None,
    mask_flag: bool = False,
    supervised_indices: torch.Tensor | None = None,
    vo_traj: PoseTrajectory | None = None,
) -> torch.Tensor:
    imudict = dataset.load_imu(sequence_index)
    N_max = imudict["u_imu"].shape[0]
    index_train_max = round(N_max * (1.0 - dataset._percent_for_val))
    batch_size = int(batch_size or dataset._batch_size)
    max_start = N_max - window_steps
    if max_start <= 0:
        raise ValueError("window_steps is longer than the sequence")

    if dataset._mode == "train":
        upper = index_train_max - window_steps
        if upper <= 0:
            raise ValueError("window_steps is longer than the training split")
        starts = torch.randint(0, upper, (batch_size,))
        starts = _filter_start_indices(
            starts, imudict, window_steps, mask_flag, supervised_indices, vo_traj
        )
        attempts = 0
        while starts.numel() < batch_size and attempts < 10:
            extra = torch.randint(0, upper, (batch_size,))
            extra = _filter_start_indices(
                extra, imudict, window_steps, mask_flag, supervised_indices, vo_traj
            )
            starts = torch.cat([starts, extra], dim=0).unique(sorted=False)
            attempts += 1
        if starts.numel() == 0:
            raise ValueError("No valid training windows found")
        return starts[:batch_size]

    if dataset._mode == "val":
        starts = torch.arange(index_train_max, max_start + 1, window_steps)
        starts = _filter_start_indices(
            starts, imudict, window_steps, mask_flag, supervised_indices, vo_traj
        )
        if starts.numel() == 0:
            raise ValueError("No valid validation windows found")
        return starts

    raise ValueError("Supervision sampling is only for train or val mode")


def build_supervised_indices(
    window_steps: int,
    sup_stride: int,
    include_window_end: bool = True,
) -> torch.Tensor:
    if window_steps < 2:
        raise ValueError("window_steps must be at least 2")
    if sup_stride <= 0:
        raise ValueError("sup_stride must be positive")
    indices = torch.arange(0, window_steps, sup_stride, dtype=torch.long)
    if include_window_end and indices[-1].item() != window_steps - 1:
        indices = torch.cat([indices, torch.tensor([window_steps - 1], dtype=torch.long)])
    if indices.numel() < 2:
        indices = torch.tensor([0, window_steps - 1], dtype=torch.long)
    return indices


def build_gt_nll_supervised_indices(
    supervised_indices: torch.Tensor,
    dense_stride: int = 4,
) -> torch.Tensor:
    """Use multi-IMU-step GT intervals for NLL while leaving sparse GT intact.

    A dense GT batch has one supervised state per IMU state, identified by
    consecutive indices.  It is decimated without appending the window end, so
    an even-sized window such as 64 yields ``[0, 4, ..., 60]`` by default and never creates
    a final one-step covariance.  GT that is already sampled at intervals of at
    least two IMU increments is returned unchanged.
    """
    if supervised_indices.dim() != 1 or supervised_indices.numel() < 2:
        raise ValueError("GT supervision indices must contain at least two states")
    if dense_stride < 2:
        raise ValueError("GT NLL dense_stride must be at least 2")

    gaps = supervised_indices[1:] - supervised_indices[:-1]
    if torch.any(gaps <= 0):
        raise ValueError("GT supervision indices must be strictly increasing")

    if torch.all(gaps == 1):
        nll_indices = supervised_indices[::dense_stride]
    else:
        if torch.any(gaps < 2):
            raise ValueError(
                "Non-dense GT NLL supervision contains a one-step interval"
            )
        nll_indices = supervised_indices

    if nll_indices.numel() < 2:
        raise ValueError("GT NLL window is too short for two-step supervision")
    if torch.any(nll_indices[1:] - nll_indices[:-1] < 2):
        raise RuntimeError("GT NLL supervision must span at least two IMU increments")
    return nll_indices


def resolve_supervision_stride(
    dt: float,
    sup_stride: int | None = None,
    sup_rate_hz: float | None = None,
) -> int:
    if sup_stride is not None:
        return int(sup_stride)
    if sup_rate_hz is None:
        raise ValueError("Either sup_stride or sup_rate_hz must be provided")
    return max(1, int(round(1.0 / (float(dt) * float(sup_rate_hz)))))


def resolve_vo_supervision_stride(
    dt: float,
    vo_traj: PoseTrajectory,
    sup_stride: int | None = None,
    sup_rate_hz: float | None = None,
) -> int:
    """Resolve the IMU stride between consecutive VO supervision samples."""
    if sup_stride is not None or sup_rate_hz is not None:
        return resolve_supervision_stride(dt, sup_stride=sup_stride, sup_rate_hz=sup_rate_hz)
    vo_dt = vo_traj.t[1:] - vo_traj.t[:-1]
    vo_dt = vo_dt[vo_dt > 0]
    if vo_dt.numel() == 0:
        raise ValueError("VO trajectory must contain at least two increasing timestamps")
    return max(1, int(round(float(torch.median(vo_dt).item()) / float(dt))))


def build_vo_supervised_indices(
    vo_window_samples: int,
    sup_stride: int,
) -> tuple[int, torch.Tensor]:
    if vo_window_samples < 2:
        raise ValueError("VO window must contain at least two supervision samples")
    if sup_stride <= 0:
        raise ValueError("sup_stride must be positive")
    imu_window_steps = (int(vo_window_samples) - 1) * int(sup_stride) + 1
    supervised_indices = torch.arange(
        0,
        imu_window_steps,
        int(sup_stride),
        dtype=torch.long,
    )
    if supervised_indices.numel() != int(vo_window_samples):
        raise RuntimeError("Failed to build the requested number of VO supervision samples")
    return imu_window_steps, supervised_indices


def read_vo_pose_file(
    file_path: str,
    sequence_start_timestamp: float | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PoseTrajectory:
    """Read VO/VIO pose text.

    Supported layouts:
    - OpenVINS-style whitespace text: timestamp px py pz qx qy qz qw
    - EuRoC-style CSV: timestamp, px, py, pz, qw, qx, qy, qz, ...
    """
    first_line = _read_first_nonempty_line(file_path)
    delimiter = "," if "," in first_line else None
    skip_header = 0 if _line_starts_with_number(first_line) else 1
    mat = np.genfromtxt(file_path, delimiter=delimiter, skip_header=skip_header, comments="#")
    if mat.ndim == 1:
        mat = mat[None, :]
    if mat.shape[1] < 8:
        raise ValueError(f"VO file must have at least 8 columns: {file_path}")
    t = normalize_pose_timestamps(mat[:, 0], sequence_start_timestamp)
    p = torch.as_tensor(mat[:, 1:4], device=device, dtype=dtype)
    q = torch.as_tensor(mat[:, 4:8], device=device, dtype=dtype)
    R = Lie.quat_to_SO3(q, order=_quat_order_from_header(first_line))
    t_tensor = torch.as_tensor(t, device=device, dtype=dtype)
    v = estimate_trajectory_velocity(t_tensor, p)
    return PoseTrajectory(t=t_tensor, R=R, p=p, v=v)


def _read_first_nonempty_line(file_path: str) -> str:
    with open(file_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                return stripped
    raise ValueError(f"VO file is empty: {file_path}")


def _line_starts_with_number(line: str) -> bool:
    token = line.lstrip("#").replace(",", " ").split()[0]
    try:
        float(token)
    except ValueError:
        return False
    return True


def _quat_order_from_header(first_line: str) -> str:
    if _line_starts_with_number(first_line):
        return "xyzw"
    header = first_line.lstrip("#").lower()
    fields = header.split(",") if "," in header else header.split()
    fields = [field.strip().replace(" ", "") for field in fields]
    if "q_rs_w" in header or (len(fields) > 4 and fields[4] in ("qw", "q_w")):
        return "wxyz"
    return "xyzw"


def load_vo_for_sequence(
    dataset,
    sequence_index: int,
    vo_dir: str,
    vo_template: str,
    vo_align: str = "first_gt",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PoseTrajectory:
    seq = dataset.sequences[sequence_index]
    file_path = os.path.join(vo_dir, vo_template.format(seq=seq))
    vo_traj = read_vo_pose_file(
        file_path,
        sequence_start_timestamp=dataset.get_start_timestamp(sequence_index),
        device=device,
        dtype=dtype,
    )
    if vo_align == "none":
        return vo_traj
    if vo_align != "first_gt":
        raise ValueError("vo_align must be one of: first_gt, none")

    gtdict = dataset.load_gt(sequence_index)
    gt_traj = PoseTrajectory(
        t=gtdict["t_gt"].to(device=device, dtype=dtype),
        R=gtdict["R_gt"].to(device=device, dtype=dtype),
        p=gtdict["p_gt"].to(device=device, dtype=dtype),
    )
    return align_vo_first_gt(vo_traj, gt_traj)


def align_vo_first_gt(vo_traj: PoseTrajectory, gt_traj: PoseTrajectory) -> PoseTrajectory:
    """Align VO to the dataset frame using the first common/nearest timestamp."""
    device = vo_traj.t.device
    dtype = vo_traj.t.dtype
    gt_t = gt_traj.t.to(device=device, dtype=dtype)
    gt_R = gt_traj.R.to(device=device, dtype=vo_traj.R.dtype)
    gt_p = gt_traj.p.to(device=device, dtype=vo_traj.p.dtype)

    start_time = torch.maximum(vo_traj.t[0], gt_t[0])
    idx_vo = torch.searchsorted(vo_traj.t, start_time).clamp(max=vo_traj.t.shape[0] - 1)
    idx_gt = torch.searchsorted(gt_t, start_time).clamp(max=gt_t.shape[0] - 1)
    R_align = torch.matmul(gt_R[idx_gt], vo_traj.R[idx_vo].transpose(-1, -2))
    p_align = gt_p[idx_gt] - torch.matmul(R_align, vo_traj.p[idx_vo].unsqueeze(-1)).squeeze(-1)
    R_aligned = torch.matmul(R_align.unsqueeze(0), vo_traj.R)
    p_aligned = torch.matmul(R_align, vo_traj.p.unsqueeze(-1)).squeeze(-1) + p_align
    v_aligned = None
    if vo_traj.v is not None:
        v_aligned = torch.matmul(R_align, vo_traj.v.unsqueeze(-1)).squeeze(-1)
    return PoseTrajectory(t=vo_traj.t, R=R_aligned, p=p_aligned, v=v_aligned)


def move_pose_trajectory(
    pose_traj: PoseTrajectory,
    device: torch.device | str,
    dtype: torch.dtype,
) -> PoseTrajectory:
    v = None
    if pose_traj.v is not None:
        v = pose_traj.v.to(device=device, dtype=dtype)
    return PoseTrajectory(
        t=pose_traj.t.to(device=device, dtype=dtype),
        R=pose_traj.R.to(device=device, dtype=dtype),
        p=pose_traj.p.to(device=device, dtype=dtype),
        v=v,
    )


def interpolate_pose(t_query: torch.Tensor, pose_traj: PoseTrajectory) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate pose trajectory at t_query with SO(3) interpolation and linear position."""
    original_shape = t_query.shape
    t_flat = t_query.reshape(-1).to(device=pose_traj.t.device, dtype=pose_traj.t.dtype)
    t_src = pose_traj.t.contiguous()
    idx1 = torch.searchsorted(t_src, t_flat).clamp(1, t_src.shape[0] - 1)
    idx0 = idx1 - 1
    t0 = t_src[idx0]
    t1 = t_src[idx1]
    alpha = ((t_flat - t0) / (t1 - t0).clamp_min(torch.finfo(t_src.dtype).eps)).clamp(0.0, 1.0)

    R0 = pose_traj.R[idx0]
    R1 = pose_traj.R[idx1]
    delta = Lie.SO3log(torch.matmul(R0.transpose(-1, -2), R1))
    R_interp = torch.matmul(R0, Lie.SO3exp(alpha.unsqueeze(-1) * delta))
    p0 = pose_traj.p[idx0]
    p1 = pose_traj.p[idx1]
    p_interp = p0 + alpha.unsqueeze(-1) * (p1 - p0)
    return R_interp.reshape(*original_shape, 3, 3), p_interp.reshape(*original_shape, 3)


def interpolate_velocity(t_query: torch.Tensor, pose_traj: PoseTrajectory) -> torch.Tensor:
    """Linearly interpolate VO-derived velocity at t_query."""
    original_shape = t_query.shape
    t_flat = t_query.reshape(-1).to(device=pose_traj.t.device, dtype=pose_traj.t.dtype)
    t_src = pose_traj.t.contiguous()
    v_src = pose_traj.v
    if v_src is None:
        v_src = estimate_trajectory_velocity(t_src, pose_traj.p)
    idx1 = torch.searchsorted(t_src, t_flat).clamp(1, t_src.shape[0] - 1)
    idx0 = idx1 - 1
    t0 = t_src[idx0]
    t1 = t_src[idx1]
    alpha = ((t_flat - t0) / (t1 - t0).clamp_min(torch.finfo(t_src.dtype).eps)).clamp(0.0, 1.0)
    v0 = v_src[idx0]
    v1 = v_src[idx1]
    v_interp = v0 + alpha.unsqueeze(-1) * (v1 - v0)
    return v_interp.reshape(*original_shape, 3)


def estimate_trajectory_velocity(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Estimate global-frame velocity from a timestamped position trajectory."""
    v = torch.zeros_like(p)
    if t.shape[0] < 2:
        return v
    eps = torch.finfo(t.dtype).eps
    dt_forward = (t[1:] - t[:-1]).unsqueeze(-1).clamp_min(eps)
    forward = (p[1:] - p[:-1]) / dt_forward
    v[0] = forward[0]
    v[-1] = forward[-1]
    if t.shape[0] > 2:
        dt_center = (t[2:] - t[:-2]).unsqueeze(-1).clamp_min(eps)
        v[1:-1] = (p[2:] - p[:-2]) / dt_center
    return v


def normalize_pose_timestamps(
    timestamps: np.ndarray,
    sequence_start_timestamp: float | None = None,
) -> np.ndarray:
    """Convert common absolute timestamp formats to sequence-relative seconds."""
    t = np.asarray(timestamps, dtype=np.float64)
    if t.shape[0] < 2:
        return t.copy()
    median_dt = np.median(np.diff(t))
    if sequence_start_timestamp is None:
        if median_dt > 1000.0:
            return (t - t[0]) / 1e9
        if t[0] > 1e6:
            return t - t[0]
        return t.copy()

    start = float(sequence_start_timestamp)
    if median_dt > 1000.0:
        return (t - start) / 1e9
    if t[0] > 1e6:
        start_sec = start / 1e9 if start > 1e12 else start
        return t - start_sec
    return t.copy()


def _filter_start_indices(
    starts: torch.Tensor,
    imudict: dict,
    window_steps: int,
    mask_flag: bool,
    supervised_indices: torch.Tensor | None,
    vo_traj: PoseTrajectory | None,
) -> torch.Tensor:
    starts = starts.to(dtype=torch.long, device=torch.device("cpu"))
    if starts.numel() == 0:
        return starts
    valid = torch.ones_like(starts, dtype=torch.bool)
    if mask_flag:
        mask_missing = imudict["mask_missing"].to(dtype=torch.bool)
        valid = valid & (~mask_missing[starts]) & (~mask_missing[starts + window_steps - 1])
    if vo_traj is not None and supervised_indices is not None:
        t_imu = imudict["t_imu"]
        supervised_indices = supervised_indices.to(dtype=torch.long, device=torch.device("cpu"))
        t_ref = t_imu[starts.unsqueeze(1) + supervised_indices.unsqueeze(0)]
        vo_t = vo_traj.t.detach().cpu().contiguous()
        in_range = (t_ref >= vo_t[0]) & (t_ref <= vo_t[-1])
        valid = valid & in_range.all(dim=1)
        window_start = t_imu[starts].contiguous()
        window_end = t_imu[starts + window_steps - 1].contiguous()
        left = torch.searchsorted(vo_t, window_start)
        right = torch.searchsorted(vo_t, window_end, right=True)
        valid = valid & ((right - left) >= 2)
    return starts[valid]
