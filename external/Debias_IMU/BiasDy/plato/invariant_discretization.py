from __future__ import annotations

import torch

try:
    import lie_algebra as Lie
except ImportError:  # Support `python -m BiasDy.plato...` from Debias_IMU root.
    from BiasDy import lie_algebra as Lie


def so3_gamma_matrices(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the SO(3) ZOH matrices Gamma_1(phi) and Gamma_2(phi).

    The convention is

        Gamma_q(phi) = sum_{n=0}^inf hat(phi)^n / (n + q)!,

    so Gamma_1 is the SO(3) left Jacobian and Gamma_2 is the second
    time-integral needed by the position update.  Polynomial small-angle
    branches keep both float32 and float64 derivatives finite at the origin.
    """
    if phi.shape[-1] != 3:
        raise ValueError(f"phi must end in 3 coordinates, got {tuple(phi.shape)}")

    a, b, d, _, _, _ = _gamma_coefficients(phi)
    K = Lie.so3hat(phi)
    K2 = torch.matmul(K, K)
    eye = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(
        *phi.shape[:-1], 3, 3
    )
    gamma1 = eye + a[..., None, None] * K + b[..., None, None] * K2
    gamma2 = 0.5 * eye + b[..., None, None] * K + d[..., None, None] * K2
    return gamma1, gamma2


def so3_gamma_action_jacobians(
    phi: torch.Tensor,
    vector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jacobians of ``Gamma_1(phi) vector`` and ``Gamma_2(phi) vector``.

    Each returned matrix has shape ``[..., 3, 3]`` and maps ``dphi`` to the
    corresponding first-order change of the gamma action.  This closed-form
    expression avoids differentiating a matrix exponential inside every IMU
    step while remaining differentiable when used by an autograd reference.
    """
    if phi.shape[-1] != 3 or vector.shape[-1] != 3:
        raise ValueError("phi and vector must both end in 3 coordinates")
    batch_shape = torch.broadcast_shapes(phi.shape[:-1], vector.shape[:-1])
    phi = phi.expand(*batch_shape, 3)
    vector = vector.expand(*batch_shape, 3)

    a, b, d, da_ds, db_ds, dd_ds = _gamma_coefficients(phi)
    K = Lie.so3hat(phi)
    Kv = torch.matmul(K, vector.unsqueeze(-1)).squeeze(-1)
    K2v = torch.matmul(K, Kv.unsqueeze(-1)).squeeze(-1)
    vector_hat = Lie.so3hat(vector)
    Kv_hat = Lie.so3hat(Kv)

    # d(Kv) = -hat(v)dphi and
    # d(K^2 v) = -hat(Kv)dphi - K hat(v)dphi.
    d_Kv = -vector_hat
    d_K2v = -Kv_hat - torch.matmul(K, vector_hat)
    grad_a = 2.0 * da_ds[..., None] * phi
    grad_b = 2.0 * db_ds[..., None] * phi
    grad_d = 2.0 * dd_ds[..., None] * phi

    D1 = (
        a[..., None, None] * d_Kv
        + b[..., None, None] * d_K2v
        + _outer(Kv, grad_a)
        + _outer(K2v, grad_b)
    )
    D2 = (
        b[..., None, None] * d_Kv
        + d[..., None, None] * d_K2v
        + _outer(Kv, grad_b)
        + _outer(K2v, grad_d)
    )
    return D1, D2


def se23_zoh_step(
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    corrected_omega: torch.Tensor,
    corrected_specific_force: torch.Tensor,
    dt: torch.Tensor | float,
    gravity: torch.Tensor | tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One structure-preserving SE_2(3) step with zero-order-held IMU input.

    Returns ``(R_next, v_next, p_next, theta)``.  Bias dynamics remain a
    separate Euler update, matching the learned continuous bias ODE used by
    this project and its inner adjoint.
    """
    batch_shape = torch.broadcast_shapes(
        R.shape[:-2],
        v.shape[:-1],
        p.shape[:-1],
        corrected_omega.shape[:-1],
        corrected_specific_force.shape[:-1],
        torch.as_tensor(dt).shape,
    )
    device = R.device
    dtype = R.dtype
    R = R.expand(*batch_shape, 3, 3)
    v = v.to(device=device, dtype=dtype).expand(*batch_shape, 3)
    p = p.to(device=device, dtype=dtype).expand(*batch_shape, 3)
    corrected_omega = corrected_omega.to(device=device, dtype=dtype).expand(
        *batch_shape, 3
    )
    corrected_specific_force = corrected_specific_force.to(
        device=device, dtype=dtype
    ).expand(*batch_shape, 3)
    dt = torch.as_tensor(dt, device=device, dtype=dtype).expand(batch_shape)
    gravity = torch.as_tensor(gravity, device=device, dtype=dtype).expand(
        *batch_shape, 3
    )

    theta = corrected_omega * dt[..., None]
    gamma1, gamma2 = so3_gamma_matrices(theta)
    dv_body = torch.matmul(
        gamma1, corrected_specific_force.unsqueeze(-1)
    ).squeeze(-1) * dt[..., None]
    dp_body = torch.matmul(
        gamma2, corrected_specific_force.unsqueeze(-1)
    ).squeeze(-1) * dt.square()[..., None]

    R_next = torch.matmul(R, Lie.SO3exp(theta))
    v_next = (
        v
        + gravity * dt[..., None]
        + torch.matmul(R, dv_body.unsqueeze(-1)).squeeze(-1)
    )
    p_next = (
        p
        + v * dt[..., None]
        + 0.5 * gravity * dt.square()[..., None]
        + torch.matmul(R, dp_body.unsqueeze(-1)).squeeze(-1)
    )
    return R_next, v_next, p_next, theta


def se23_zoh_bias_input_matrix(
    R: torch.Tensor,
    v_next: torch.Tensor,
    p_next: torch.Tensor,
    corrected_omega: torch.Tensor,
    corrected_specific_force: torch.Tensor,
    dt: torch.Tensor | float,
) -> torch.Tensor:
    """Exact discrete mean-state Jacobian with respect to ``[b_a, b_g]``.

    The output tangent uses the project's left perturbation convention
    ``X(db) = Exp(B db) X`` and is ordered ``[phi, rho_v, rho_p]``.  Unlike
    the paper's first-order stochastic ``G_k``, this finite-step matrix also
    contains the O(dt^2) BA-to-position and gyro/acceleration coupling terms.
    Its first-order expansion is exactly ``se23_Gd``.
    """
    batch_shape = torch.broadcast_shapes(
        R.shape[:-2],
        v_next.shape[:-1],
        p_next.shape[:-1],
        corrected_omega.shape[:-1],
        corrected_specific_force.shape[:-1],
        torch.as_tensor(dt).shape,
    )
    device = R.device
    dtype = R.dtype
    R = R.expand(*batch_shape, 3, 3)
    v_next = v_next.to(device=device, dtype=dtype).expand(*batch_shape, 3)
    p_next = p_next.to(device=device, dtype=dtype).expand(*batch_shape, 3)
    corrected_omega = corrected_omega.to(device=device, dtype=dtype).expand(
        *batch_shape, 3
    )
    corrected_specific_force = corrected_specific_force.to(
        device=device, dtype=dtype
    ).expand(*batch_shape, 3)
    dt = torch.as_tensor(dt, device=device, dtype=dtype).expand(batch_shape)

    theta = corrected_omega * dt[..., None]
    gamma1, gamma2 = so3_gamma_matrices(theta)
    Dgamma1, Dgamma2 = so3_gamma_action_jacobians(
        theta, corrected_specific_force
    )

    C1 = dt[..., None, None] * gamma1
    C2 = dt.square()[..., None, None] * gamma2
    H = -torch.matmul(R, C1)
    raw_v_bg = -dt.square()[..., None, None] * torch.matmul(R, Dgamma1)
    raw_p_bg = -dt.pow(3)[..., None, None] * torch.matmul(R, Dgamma2)

    B = torch.zeros(*batch_shape, 9, 6, device=device, dtype=dtype)
    B[..., 0:3, 3:6] = H
    B[..., 3:6, 0:3] = -torch.matmul(R, C1)
    B[..., 6:9, 0:3] = -torch.matmul(R, C2)
    B[..., 3:6, 3:6] = raw_v_bg + torch.matmul(
        Lie.so3hat(v_next), H
    )
    B[..., 6:9, 3:6] = raw_p_bg + torch.matmul(
        Lie.so3hat(p_next), H
    )
    return B


def _gamma_coefficients(
    phi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a, b, d and their derivatives with respect to s=||phi||^2."""
    s = (phi * phi).sum(dim=-1)
    # The fourth-order polynomials are much more accurate than the quotient
    # formula near zero, especially for d/ds in float32.
    # The retained series terms are accurate through ordinary IMU increments
    # (theta < 0.1 rad) and avoid float32 cancellation in Gamma_2.
    small = s < 1e-2
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s

    a_small = 0.5 - s / 24.0 + s2 / 720.0 - s3 / 40320.0 + s4 / 3628800.0
    b_small = 1.0 / 6.0 - s / 120.0 + s2 / 5040.0 - s3 / 362880.0 + s4 / 39916800.0
    d_small = 1.0 / 24.0 - s / 720.0 + s2 / 40320.0 - s3 / 3628800.0 + s4 / 479001600.0
    da_small = -1.0 / 24.0 + s / 360.0 - s2 / 13440.0 + s3 / 907200.0
    db_small = -1.0 / 120.0 + s / 2520.0 - s2 / 120960.0 + s3 / 9979200.0
    dd_small = -1.0 / 720.0 + s / 20160.0 - s2 / 1209600.0 + s3 / 119750400.0

    # Clamp only the inactive general branch so torch.where never observes a
    # zero denominator during backward.
    s_safe = torch.clamp(s, min=1e-12)
    theta = torch.sqrt(s_safe)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    one_minus_cos = 1.0 - cos_theta
    theta_minus_sin = theta - sin_theta
    numerator_d = s_safe + 2.0 * cos_theta - 2.0

    a_general = one_minus_cos / s_safe
    b_general = theta_minus_sin / (s_safe * theta)
    d_general = numerator_d / (2.0 * s_safe.square())
    da_general = (
        0.5 * sin_theta * theta - one_minus_cos
    ) / s_safe.square()
    db_general = (
        0.5 * (1.0 - cos_theta) / (s_safe.square())
        - 1.5 * theta_minus_sin / (s_safe.square() * theta)
    )
    dd_general = (
        0.5 * (1.0 - sin_theta / theta) / s_safe.square()
        - numerator_d / s_safe.pow(3)
    )

    return (
        torch.where(small, a_small, a_general),
        torch.where(small, b_small, b_general),
        torch.where(small, d_small, d_general),
        torch.where(small, da_small, da_general),
        torch.where(small, db_small, db_general),
        torch.where(small, dd_small, dd_general),
    )


def _outer(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left.unsqueeze(-1) * right.unsqueeze(-2)
