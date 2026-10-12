"""Batch independent native-grid sequences using the production ZOH rollout."""
from __future__ import annotations

import torch
import Interpolation
from .rollout import integrate_batch_paper_zoh


@torch.inference_mode()
def rollout_native_sequences(dataset, model, device: str, bias_init: str = "first") -> list[dict]:
    """Return complete trajectories; shorter rows freeze after their final state.

    Requires matching native time-grid prefixes and bias dynamics with local
    IMU inputs. IMU interpolation is evaluated once on the exact forward clock.
    No integration equation is duplicated from the training rollout.
    """
    if bias_init not in ("first", "zero"):
        raise ValueError("bias_init must be first or zero")
    if getattr(model.biasfunc_a, "uses_acc_window", False):
        raise ValueError("Batched inference requires local-input bias dynamics")
    dtype = next(model.parameters()).dtype
    inputs = [dataset.load_imu(i) for i in range(dataset.length())]
    trajectories = [dataset.get_full_trajectory(i, device=device) for i in range(dataset.length())]
    counts = [len(trajectory[3]) for trajectory in trajectories]
    grid = trajectories[counts.index(max(counts))][3].detach().cpu()
    for trajectory, count in zip(trajectories, counts):
        if not torch.equal(trajectory[3].cpu(), grid[:count]):
            raise ValueError("Batched native inference requires identical time-grid prefixes")
    # Match the actual solver's accumulated absolute-time coordinate, including
    # float32 rounding, rather than substituting nominal sample timestamps.
    clock = torch.empty(len(grid), dtype=dtype)
    clock[0] = grid[0]
    increments = (grid[1:]-grid[:-1]).to(dtype)
    for k in range(len(grid)-1):
        clock[k+1] = clock[k] + increments[k]
    measurements, derivatives, initials, rotations = [], [], [], []
    for index, (imu, trajectory, count) in enumerate(zip(inputs, trajectories, counts)):
        _, raw_u, X_gt, _ = trajectory
        X0 = X_gt[0].to(dtype)
        bg0 = imu["w_bias"][0].to(device=device, dtype=dtype)
        ba0 = imu["a_bias"][0].to(device=device, dtype=dtype)
        if bias_init == "zero":
            bg0, ba0 = torch.zeros_like(bg0), torch.zeros_like(ba0)
        initials.append(torch.cat([torch.zeros(3, device=device, dtype=dtype), X0.new_tensor([clock[0]]), bg0, X0[:3,3], X0[:3,4], ba0]))
        rotations.append(X0[:3,:3])
        spline = Interpolation.CubicHermiteSpline(imu["t_imu"], imu["coeff"], device=device)
        queried = clock[:count].to(device)
        u = spline.evaluate(queried).to(dtype)
        du = spline.derivative(queried).to(dtype)
        extra = len(grid)-count
        if extra:
            u = torch.cat([u, u[-1:].expand(extra,6)])
            du = torch.cat([du, du[-1:].expand(extra,6)])
        measurements.append(u[:-1])
        derivatives.append(du[:-1])
    y0, R0 = torch.stack(initials), torch.stack(rotations)
    model.set_R0(R0)
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    if hasattr(model.biasfunc_a, "set_initial_ba"):
        model.biasfunc_a.set_initial_ba(y0[:,13:16])
    if hasattr(model.biasfunc_w, "set_initial_ba"):
        model.biasfunc_w.set_initial_ba(y0[:,13:16])
    solution, R = integrate_batch_paper_zoh(model, y0, R0, grid.to(device), valid_lengths=torch.tensor(counts, device=device), sampled_imu=(torch.stack(measurements, dim=1), torch.stack(derivatives, dim=1)))
    results = []
    for index, (trajectory, count) in enumerate(zip(trajectories, counts)):
        _, raw_u, X_gt, t_gt = trajectory
        prediction = solution[:count,index]
        results.append({"sol_pred": prediction, "R_pred": R[:count,index], "X_gt": X_gt, "t_gt": t_gt, "net_imu": raw_u - torch.cat([prediction[:,4:7],prediction[:,13:16]], dim=-1), "imu_raw": raw_u, "timestamp_start": dataset.get_start_timestamp(index), "mean_discretization": "paper_zoh"})
    return results
