from __future__ import annotations

import torch


def accumulate_bg_euler_adjoint_gradients(
    model,
    y_solution: torch.Tensor,
    t_odeint: torch.Tensor,
    jump_grad_bg: torch.Tensor,
    jump_indices: torch.Tensor | None = None,
    parameters: list[torch.nn.Parameter] | None = None,
) -> torch.Tensor:
    """Accumulate BG-network gradients with a fixed-step Euler adjoint."""
    if y_solution.ndim < 3 or y_solution.shape[-1] != 16:
        raise ValueError(f"y_solution must have shape (T, ..., 16), got {tuple(y_solution.shape)}")
    if t_odeint.ndim != 1 or t_odeint.shape[0] != y_solution.shape[0]:
        raise ValueError("t_odeint must be one-dimensional and match y_solution time length")
    if jump_grad_bg.shape[-1] != 3:
        raise ValueError(f"jump_grad_bg must have last dimension 3, got {tuple(jump_grad_bg.shape)}")
    if model.biasfunc_w is None:
        raise ValueError("model.biasfunc_w is required for BG adjoint gradients")

    T = y_solution.shape[0]
    device = y_solution.device
    dtype = y_solution.dtype
    if jump_indices is None:
        jump_indices = torch.arange(jump_grad_bg.shape[0], device=device)
    else:
        jump_indices = jump_indices.to(device=device)
    if jump_indices.ndim != 1 or jump_indices.shape[0] != jump_grad_bg.shape[0]:
        raise ValueError("jump_indices must be one-dimensional and match jump_grad_bg length")
    if jump_indices.numel() and (jump_indices.min() < 0 or jump_indices.max() >= T):
        raise ValueError("jump_indices contains an index outside the forward grid")

    jumps = torch.zeros(T, *y_solution.shape[1:-1], 3, device=device, dtype=dtype)
    jumps.index_add_(0, jump_indices.long(), jump_grad_bg.to(device=device, dtype=dtype))

    if parameters is None:
        parameters = [p for p in model.biasfunc_w.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable BG parameters were provided")

    lambda_bg = jumps[-1].detach()
    for k in range(T - 2, -1, -1):
        lambda_next = lambda_bg.detach()
        dt = (t_odeint[k + 1] - t_odeint[k]).to(device=device, dtype=dtype)
        y_k_const = y_solution[k].detach()
        bg_k = y_k_const[..., 4:7].clone().detach().requires_grad_(True)
        y_k = torch.cat([y_k_const[..., :4], bg_k, y_k_const[..., 7:]], dim=-1)

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
        f_bg = model.biasfunc_w(y_k, imu_meas, imu_meas_dot, model.R0)
        scalar = dt * (f_bg * lambda_next).sum()
        grads = torch.autograd.grad(
            scalar,
            [bg_k, *parameters],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        grad_bg = grads[0]
        if grad_bg is None:
            grad_bg = torch.zeros_like(lambda_next)
        for parameter, grad in zip(parameters, grads[1:]):
            if grad is None:
                continue
            if parameter.grad is None:
                parameter.grad = grad.detach().clone()
            else:
                parameter.grad.add_(grad.detach())

        lambda_bg = lambda_next + grad_bg.detach()
        lambda_bg = lambda_bg + jumps[k].detach()

    return lambda_bg
