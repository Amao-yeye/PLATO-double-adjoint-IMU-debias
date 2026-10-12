from __future__ import annotations

import torch


def accumulate_ba_euler_adjoint_gradients(
    model,
    y_solution: torch.Tensor,
    t_odeint: torch.Tensor,
    jump_grad_ba: torch.Tensor,
    jump_indices: torch.Tensor | None = None,
    parameters: list[torch.nn.Parameter] | None = None,
) -> torch.Tensor:
    """Accumulate BA-network gradients with a fixed-step Euler adjoint.

    The full IMU rollout is detached. PyTorch autograd is used only for the
    local vector-Jacobian product of the BA network at each time step.
    """
    if y_solution.ndim < 3 or y_solution.shape[-1] != 16:
        raise ValueError(f"y_solution must have shape (T, ..., 16), got {tuple(y_solution.shape)}")
    if t_odeint.ndim != 1 or t_odeint.shape[0] != y_solution.shape[0]:
        raise ValueError("t_odeint must be one-dimensional and match y_solution time length")
    if jump_grad_ba.shape[-1] != 3:
        raise ValueError(f"jump_grad_ba must have last dimension 3, got {tuple(jump_grad_ba.shape)}")
    if model.biasfunc_a is None:
        raise ValueError("model.biasfunc_a is required for BA adjoint gradients")

    T = y_solution.shape[0]
    device = y_solution.device
    dtype = y_solution.dtype
    if jump_indices is None:
        jump_indices = torch.arange(jump_grad_ba.shape[0], device=device)
    else:
        jump_indices = jump_indices.to(device=device)
    if jump_indices.ndim != 1 or jump_indices.shape[0] != jump_grad_ba.shape[0]:
        raise ValueError("jump_indices must be one-dimensional and match jump_grad_ba length")
    if jump_indices.numel() and (jump_indices.min() < 0 or jump_indices.max() >= T):
        raise ValueError("jump_indices contains an index outside the forward grid")

    jumps = torch.zeros(T, *y_solution.shape[1:-1], 3, device=device, dtype=dtype)
    jumps.index_add_(0, jump_indices.long(), jump_grad_ba.to(device=device, dtype=dtype))

    if parameters is None:
        parameters = [p for p in model.biasfunc_a.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable BA parameters were provided")

    lambda_ba = jumps[-1].detach()
    for k in range(T - 2, -1, -1):
        lambda_next = lambda_ba.detach()
        dt = (t_odeint[k + 1] - t_odeint[k]).to(device=device, dtype=dtype)
        y_k_const = y_solution[k].detach()
        ba_k = y_k_const[..., 13:16].clone().detach().requires_grad_(True)
        y_k = torch.cat([y_k_const[..., :13], ba_k], dim=-1)

        t_abs = y_k_const[..., 3]
        # Match VFSE23vbiasBase.__call__: cached spline coefficients may be
        # float64 even when the checkpoint network and nominal rollout are
        # float32.  The local adjoint VJP must replay the exact forward dtype.
        imu_meas = model.u_func(t_abs).detach().to(
            device=y_k.device, dtype=y_k.dtype
        )
        imu_meas_dot = model.u_dot_func(t_abs).detach().to(
            device=y_k.device, dtype=y_k.dtype
        )
        f_ba = model.biasfunc_a(y_k, imu_meas, imu_meas_dot, model.R0)
        scalar = dt * (f_ba * lambda_next).sum()
        grads = torch.autograd.grad(
            scalar,
            [ba_k, *parameters],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        grad_ba = grads[0]
        if grad_ba is None:
            grad_ba = torch.zeros_like(lambda_next)
        for parameter, grad in zip(parameters, grads[1:]):
            if grad is None:
                continue
            if parameter.grad is None:
                parameter.grad = grad.detach().clone()
            else:
                parameter.grad.add_(grad.detach())

        lambda_ba = lambda_next + grad_ba.detach()
        lambda_ba = lambda_ba + jumps[k].detach()

    return lambda_ba
