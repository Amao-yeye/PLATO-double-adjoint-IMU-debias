"""Shared training and inference rollout; IMU ordering is [gyro, accel].

State-vector ordering is [chart(3), absolute time, bg(3), v(3), p(3), ba(3)].
Rotation maps body coordinates to world coordinates. Bias ODEs use Euler;
rotation, velocity and position use exact zero-order-held input propagation.
"""
from __future__ import annotations

import torch

from .invariant_discretization import se23_zoh_step


def integrate_batch_paper_zoh(
    model,
    y0: torch.Tensor,
    R0: torch.Tensor,
    t_odeint: torch.Tensor,
    *,
    valid_lengths: torch.Tensor | None = None,
    sampled_imu: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out Eq. (1) with a structure-preserving ZOH SE_2(3) step.

    The IMU and learned bias rates are evaluated at the left endpoint. Biases
    therefore retain the fixed-step Euler update assumed by the inner
    adjoints, while R/v/p use the exact constant-input Lie-group solution.
    """
    if t_odeint.ndim != 1 or t_odeint.shape[0] < 2:
        raise ValueError("t_odeint must be one-dimensional with at least two entries")
    if valid_lengths is not None:
        valid_lengths = torch.as_tensor(valid_lengths, device=y0.device)
        if y0.ndim != 2 or valid_lengths.shape != y0.shape[:1]:
            raise ValueError("valid_lengths requires a one-dimensional length per batch row")
        if valid_lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("valid_lengths must contain integer state counts")
        if torch.any(valid_lengths < 2) or torch.any(valid_lengths > len(t_odeint)):
            raise ValueError("valid_lengths must be between 2 and the common grid length")
    if sampled_imu is not None:
        expected = (len(t_odeint)-1, *y0.shape[:-1], 6)
        if len(sampled_imu) != 2 or any(tuple(value.shape) != expected for value in sampled_imu):
            raise ValueError(f"sampled_imu must contain measurement and derivative tensors of shape {expected}")
        if getattr(model.biasfunc_a, "uses_acc_window", False):
            raise ValueError("Precomputed inference inputs do not support historical IMU-window networks")

    solution_steps = [y0]
    rotation_steps = [R0]
    y_current = y0
    R_current = R0
    gravity = model.g_const.to(device=y0.device, dtype=y0.dtype)

    for step in range(t_odeint.shape[0] - 1):
        dt = (t_odeint[step + 1] - t_odeint[step]).to(
            device=y0.device, dtype=y0.dtype
        )
        if valid_lengths is not None:
            dt = dt * (step < valid_lengths-1).to(y0.dtype)
        dt_bias = dt.unsqueeze(-1) if valid_lengths is not None else dt
        y_eval = torch.cat(
            [torch.zeros_like(y_current[..., :3]), y_current[..., 3:]], dim=-1
        )
        (
            corrected_omega,
            corrected_specific_force,
            bg_dot,
            ba_dot,
        ) = evaluate_corrected_imu_and_bias_rates(
            model, y_eval, R_current,
            imu_meas=None if sampled_imu is None else sampled_imu[0][step],
            imu_meas_dot=None if sampled_imu is None else sampled_imu[1][step],
        )

        R_next, v_next, p_next, theta = se23_zoh_step(
            R_current,
            y_eval[..., 7:10],
            y_eval[..., 10:13],
            corrected_omega,
            corrected_specific_force,
            dt,
            gravity=gravity,
        )
        y_next = torch.cat(
            [
                theta,
                y_eval[..., 3:4] + dt_bias,
                y_eval[..., 4:7] + dt_bias * bg_dot,
                v_next,
                p_next,
                y_eval[..., 13:16] + dt_bias * ba_dot,
            ],
            dim=-1,
        )
        solution_steps.append(y_next)
        rotation_steps.append(R_next)
        y_current = y_next
        R_current = R_next

    # Do not leave a rollout graph attached to this stateful compatibility
    # attribute. Bias networks received R_current explicitly above.
    model.set_R0(rotation_steps[-2].detach())
    return torch.stack(solution_steps, dim=0), torch.stack(rotation_steps, dim=0)


def evaluate_corrected_imu_and_bias_rates(
    model,
    y: torch.Tensor,
    R: torch.Tensor,
    *,
    imu_meas: torch.Tensor | None = None,
    imu_meas_dot: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate held IMU input and the two learned bias-rate ODEs."""
    t_abs = y[..., 3]
    imu_meas = (model.u_func(t_abs) if imu_meas is None else imu_meas).to(device=y.device, dtype=y.dtype)
    imu_meas_dot = (model.u_dot_func(t_abs) if imu_meas_dot is None else imu_meas_dot).to(device=y.device, dtype=y.dtype)

    if model._set_bw_zero:
        bg = torch.zeros_like(y[..., 4:7])
        bg_dot = torch.zeros_like(bg)
    else:
        if model.biasfunc_w is None:
            raise ValueError("biasfunc_w is required when gyro bias is enabled")
        bg = y[..., 4:7]
        bg_dot = model.biasfunc_w(y, imu_meas, imu_meas_dot, R)

    if model._set_ba_zero:
        ba = torch.zeros_like(y[..., 13:16])
        ba_dot = torch.zeros_like(ba)
    else:
        if model.biasfunc_a is None:
            raise ValueError("biasfunc_a is required when accelerometer bias is enabled")
        ba = y[..., 13:16]
        if getattr(model.biasfunc_a, "uses_acc_window", False):
            acc_window = model._evaluate_acc_window(
                t_abs, getattr(model.biasfunc_a, "acc_window_steps", 1)
            )
            ba_dot = model.biasfunc_a(y, imu_meas, imu_meas_dot, R, acc_window)
        else:
            ba_dot = model.biasfunc_a(y, imu_meas, imu_meas_dot, R)

    return imu_meas[..., :3] - bg, imu_meas[..., 3:] - ba, bg_dot, ba_dot
