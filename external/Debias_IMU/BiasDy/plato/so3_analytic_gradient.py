from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import lie_algebra as Lie
except ImportError:  # Support `python -m BiasDy.plato...` from Debias_IMU root.
    from BiasDy import lie_algebra as Lie

try:
    from plato.invariant_discretization import so3_gamma_matrices
except ImportError:
    from .invariant_discretization import so3_gamma_matrices


@dataclass
class GyroBiasAdjointGradient:
    lambda_state: torch.Tensor
    bias_gradient: torch.Tensor
    state_gradient: torch.Tensor
    residual: torch.Tensor


def so3_invariant_residual(
    R_pred: torch.Tensor,
    R_ref: torch.Tensor,
    invariant_error: str = "right",
) -> torch.Tensor:
    """Return SO(3) residuals in tangent coordinates."""
    _validate_time_major("R_pred", R_pred, trailing_shape=(3, 3), min_length=1)
    _validate_time_major("R_ref", R_ref, trailing_shape=(3, 3), min_length=1)
    if R_pred.shape != R_ref.shape:
        raise ValueError("R_pred and R_ref must have the same shape")
    if invariant_error == "right":
        return Lie.SO3log(R_ref @ R_pred.transpose(-1, -2))
    if invariant_error == "left":
        return Lie.SO3log(R_pred.transpose(-1, -2) @ R_ref)
    raise ValueError("invariant_error must be right or left")


def weighted_so3_mse_state_gradient(
    residual: torch.Tensor,
    rot_weight: float = 1.0,
    residual_jacobian: str = "bch",
) -> torch.Tensor:
    """Analytic dL/dtheta for 0.5 * mean(||w r||^2), excluding 1e6 scale."""
    _validate_time_major("residual", residual, trailing_shape=(3,), min_length=1)
    state_count = 1
    for size in residual.shape[:-1]:
        state_count *= int(size)
    grad_residual = float(rot_weight) ** 2 * residual / max(state_count, 1)
    return apply_so3_log_residual_jacobian_transpose(
        residual,
        grad_residual,
        residual_jacobian=residual_jacobian,
    )


def apply_so3_log_residual_jacobian_transpose(
    residual: torch.Tensor,
    grad_residual: torch.Tensor,
    residual_jacobian: str = "bch",
) -> torch.Tensor:
    if residual_jacobian == "identity":
        return grad_residual
    if residual_jacobian == "bch":
        H = -Lie.SO3rightJacoInv(residual)
        return (H.transpose(-1, -2) @ grad_residual.unsqueeze(-1)).squeeze(-1)
    if residual_jacobian == "left":
        H = Lie.SO3leftJacoInv(residual)
    elif residual_jacobian == "right":
        H = Lie.SO3rightJacoInv(residual)
    else:
        raise ValueError("residual_jacobian must be one of: bch, identity, left, right")
    return (H.transpose(-1, -2) @ grad_residual.unsqueeze(-1)).squeeze(-1)


def so3_gyro_bias_gradient(
    R_pred: torch.Tensor,
    R_ref: torch.Tensor,
    t: torch.Tensor | None = None,
    dt: float | torch.Tensor | None = None,
    rot_weight: float = 1.0,
    residual_jacobian: str = "bch",
    invariant_error: str = "right",
    detach: bool = True,
    discretization: str = "paper_first_order",
    corrected_omega: torch.Tensor | None = None,
    supervised_indices: torch.Tensor | None = None,
) -> GyroBiasAdjointGradient:
    """Compute analytic dL/db_g,k for right-invariant SO(3) error."""
    _validate_time_major("R_pred", R_pred, trailing_shape=(3, 3), min_length=2)
    _validate_time_major("R_ref", R_ref, trailing_shape=(3, 3), min_length=1)
    if supervised_indices is None:
        if R_pred.shape != R_ref.shape:
            raise ValueError(
                "R_pred and R_ref must have the same shape when "
                "supervised_indices is omitted"
            )
        residual = so3_invariant_residual(
            R_pred, R_ref, invariant_error=invariant_error
        )
        state_grad = weighted_so3_mse_state_gradient(
            residual,
            rot_weight=rot_weight,
            residual_jacobian=residual_jacobian,
        )
    else:
        supervised_indices = torch.as_tensor(
            supervised_indices, device=R_pred.device, dtype=torch.long
        )
        if supervised_indices.ndim != 1 or supervised_indices.numel() < 1:
            raise ValueError("supervised_indices must be a non-empty vector")
        if (
            supervised_indices.min() < 0
            or supervised_indices.max() >= R_pred.shape[0]
        ):
            raise ValueError("supervised_indices contains an out-of-range index")
        R_pred_supervised = R_pred[supervised_indices]
        if R_pred_supervised.shape != R_ref.shape:
            raise ValueError(
                "R_ref must match R_pred[supervised_indices], got "
                f"{tuple(R_ref.shape)} and {tuple(R_pred_supervised.shape)}"
            )
        residual = so3_invariant_residual(
            R_pred_supervised,
            R_ref,
            invariant_error=invariant_error,
        )
        supervised_state_grad = weighted_so3_mse_state_gradient(
            residual,
            rot_weight=rot_weight,
            residual_jacobian=residual_jacobian,
        )
        state_grad = torch.zeros(
            R_pred.shape[0],
            *R_pred.shape[1:-2],
            3,
            device=R_pred.device,
            dtype=R_pred.dtype,
        )
        state_grad.index_add_(
            0, supervised_indices, supervised_state_grad
        )
    dt_transitions = _transition_dt(R_pred.shape[0], R_pred.device, R_pred.dtype, t=t, dt=dt)
    if dt_transitions.ndim == 1 and R_pred.ndim > 3:
        dt_transitions = dt_transitions.reshape(dt_transitions.shape[0], *([1] * (R_pred.ndim - 3)))

    lambda_state = torch.empty_like(state_grad)
    lambda_state[-1] = state_grad[-1]
    for k in range(R_pred.shape[0] - 2, -1, -1):
        lambda_state[k] = state_grad[k] + lambda_state[k + 1]

    bias_gradient = torch.empty(
        R_pred.shape[0] - 1,
        *R_pred.shape[1:-2],
        3,
        device=R_pred.device,
        dtype=R_pred.dtype,
    )
    if discretization == "paper_zoh":
        expected_shape = (R_pred.shape[0] - 1, *R_pred.shape[1:-2], 3)
        if corrected_omega is None or tuple(corrected_omega.shape) != expected_shape:
            got = None if corrected_omega is None else tuple(corrected_omega.shape)
            raise ValueError(
                f"paper_zoh corrected_omega must have shape {expected_shape}, got {got}"
            )
    elif discretization != "paper_first_order":
        raise ValueError("discretization must be paper_first_order or paper_zoh")

    for k in range(R_pred.shape[0] - 1):
        dt_k = dt_transitions[k]
        if discretization == "paper_zoh":
            theta_k = corrected_omega[k] * dt_k.unsqueeze(-1)
            gamma1_k, _ = so3_gamma_matrices(theta_k)
            B_k = -torch.matmul(
                R_pred[k], dt_k.unsqueeze(-1).unsqueeze(-1) * gamma1_k
            )
            bias_gradient[k] = (
                B_k.transpose(-1, -2) @ lambda_state[k + 1].unsqueeze(-1)
            ).squeeze(-1)
        else:
            while dt_k.ndim < lambda_state[k + 1].ndim:
                dt_k = dt_k.unsqueeze(-1)
            bias_gradient[k] = -dt_k * (
                R_pred[k].transpose(-1, -2) @ lambda_state[k + 1].unsqueeze(-1)
            ).squeeze(-1)

    if detach:
        lambda_state = lambda_state.detach()
        bias_gradient = bias_gradient.detach()
        state_grad = state_grad.detach()
        residual = residual.detach()

    return GyroBiasAdjointGradient(
        lambda_state=lambda_state,
        bias_gradient=bias_gradient,
        state_gradient=state_grad,
        residual=residual,
    )


def _transition_dt(
    time_length: int,
    device: torch.device,
    dtype: torch.dtype,
    t: torch.Tensor | None = None,
    dt: float | torch.Tensor | None = None,
) -> torch.Tensor:
    if (t is None) == (dt is None):
        raise ValueError("Provide exactly one of t or dt")
    if t is not None:
        t = torch.as_tensor(t, device=device, dtype=dtype)
        if t.shape[0] != time_length:
            raise ValueError(f"t must have length {time_length}, got {t.shape[0]}")
        return t[1:] - t[:-1]

    dt_tensor = torch.as_tensor(dt, device=device, dtype=dtype)
    if dt_tensor.ndim == 0:
        return dt_tensor
    if dt_tensor.shape[0] != time_length - 1:
        raise ValueError(f"dt must be scalar or have length {time_length - 1}, got {tuple(dt_tensor.shape)}")
    return dt_tensor


def _validate_time_major(
    name: str,
    tensor: torch.Tensor,
    trailing_shape: tuple[int, ...],
    min_length: int = 1,
) -> None:
    if tensor.ndim < len(trailing_shape) + 1:
        raise ValueError(f"{name} must have a leading time dimension and trailing shape {trailing_shape}")
    if tuple(tensor.shape[-len(trailing_shape):]) != trailing_shape:
        raise ValueError(f"{name} must have trailing shape {trailing_shape}, got {tuple(tensor.shape)}")
    if tensor.shape[0] < min_length:
        raise ValueError(f"{name} must have at least {min_length} time steps")
