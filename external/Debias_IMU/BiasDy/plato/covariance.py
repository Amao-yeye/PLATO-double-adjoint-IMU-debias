from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import lie_algebra as Lie
except ImportError:  # Support `python -m BiasDy.plato...` from Debias_IMU root.
    from BiasDy import lie_algebra as Lie


@dataclass(frozen=True)
class IntervalCovarianceForwardSensitivity:
    """Nominal interval covariance and its explicit forward sensitivities.

    ``noise`` is ordered ``[direction, *batch, 9, 9]``. ``state`` is ordered
    ``[local_step, tangent_direction, *batch, 9, 9]`` and uses left-invariant
    state perturbations ``[phi, rho_v, rho_p]``.
    """

    Phi: torch.Tensor
    covariance: torch.Tensor
    noise: torch.Tensor | None
    state: torch.Tensor | None


@dataclass(frozen=True)
class SupervisionCovarianceForwardSensitivity:
    """Forward sensitivities for all adjacent supervision intervals.

    ``noise`` is ordered ``[direction, interval, *batch, 9, 9]``. State
    sensitivities are kept as one tensor per interval because supervision
    intervals need not have equal lengths.
    """

    Phi: torch.Tensor
    covariance: torch.Tensor
    noise: torch.Tensor | None
    state: tuple[torch.Tensor, ...] | None


def se23_phi(
    dt: torch.Tensor | float,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    gravity: torch.Tensor | tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> torch.Tensor:
    """SE_2(3) invariant-error transition matrix for constant gravity dynamics."""
    dt = _as_tensor(dt, device=device, dtype=dtype)
    device = dt.device
    dtype = dt.dtype
    batch_shape = dt.shape

    eye3 = _eye(3, batch_shape, device, dtype)
    phi = torch.zeros(*batch_shape, 9, 9, device=device, dtype=dtype)
    phi[..., 0:3, 0:3] = eye3
    phi[..., 3:6, 3:6] = eye3
    phi[..., 6:9, 6:9] = eye3

    g = torch.as_tensor(gravity, device=device, dtype=dtype).expand(*batch_shape, 3)
    g_hat = Lie.so3hat(g)
    dt_mat = dt.unsqueeze(-1).unsqueeze(-1)
    phi[..., 3:6, 0:3] = g_hat * dt_mat
    phi[..., 6:9, 0:3] = 0.5 * g_hat * dt_mat.square()
    phi[..., 6:9, 3:6] = eye3 * dt_mat
    return phi


def se23_Gd(
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    dt: torch.Tensor | float,
) -> torch.Tensor:
    """Discrete IMU noise injection matrix.

    The convention is per-sample discrete noise:
    Q_d,k = G_d,k Q_n G_d,k^T, where Q_n = diag(sigma_a^2 I, sigma_g^2 I).
    Columns are ordered [n_a, n_g], rows [phi, rho_v, rho_p].
    """
    device = R.device
    dtype = R.dtype
    dt = _as_tensor(dt, device=device, dtype=dtype)
    batch_shape = torch.broadcast_shapes(
        R.shape[:-2], v.shape[:-1], p.shape[:-1], dt.shape
    )

    R = R.expand(*batch_shape, 3, 3)
    v = v.expand(*batch_shape, 3)
    p = p.expand(*batch_shape, 3)
    dt = dt.expand(batch_shape)

    zeros = torch.zeros(*batch_shape, 3, 3, device=device, dtype=dtype)
    G = torch.zeros(*batch_shape, 9, 6, device=device, dtype=dtype)
    G[..., 0:3, 0:3] = zeros
    G[..., 0:3, 3:6] = R
    G[..., 3:6, 0:3] = R
    G[..., 3:6, 3:6] = torch.matmul(Lie.so3hat(v), R)
    G[..., 6:9, 0:3] = zeros
    G[..., 6:9, 3:6] = torch.matmul(Lie.so3hat(p), R)
    return -dt.unsqueeze(-1).unsqueeze(-1) * G


def se23_Gd_left_tangent_sensitivity(
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    dt: torch.Tensor | float,
) -> torch.Tensor:
    """Differentiate ``G_d(R, v, p)`` along the nine left SE_2(3) directions.

    For ``X(delta) = Exp(delta) X`` at ``delta=0``:

    ``dR = hat(phi) R``, ``dv = hat(phi) v + rho_v``, and
    ``dp = hat(phi) p + rho_p``.

    The returned tensor has shape ``[*batch, 9, 9, 6]``; its third-to-last
    axis indexes tangent directions ``[phi, rho_v, rho_p]``.  This derivative
    is required because the invariant transition ``F_k`` is state-independent
    in this model while ``G_k`` is not.
    """
    device = R.device
    dtype = R.dtype
    dt = _as_tensor(dt, device=device, dtype=dtype)
    batch_shape = torch.broadcast_shapes(
        R.shape[:-2], v.shape[:-1], p.shape[:-1], dt.shape
    )

    R = R.expand(*batch_shape, 3, 3)
    v = v.expand(*batch_shape, 3)
    p = p.expand(*batch_shape, 3)
    dt = dt.expand(batch_shape)

    tangent_basis = torch.eye(9, device=device, dtype=dtype)
    tangent_basis = tangent_basis.expand(*batch_shape, 9, 9)
    phi = tangent_basis[..., 0:3]
    rho_v = tangent_basis[..., 3:6]
    rho_p = tangent_basis[..., 6:9]

    phi_hat = Lie.so3hat(phi)
    R_for_directions = R.unsqueeze(-3)
    dR = torch.matmul(phi_hat, R_for_directions)
    dv = (
        torch.matmul(phi_hat, v.unsqueeze(-2).unsqueeze(-1)).squeeze(-1)
        + rho_v
    )
    dp = (
        torch.matmul(phi_hat, p.unsqueeze(-2).unsqueeze(-1)).squeeze(-1)
        + rho_p
    )

    dG = torch.zeros(*batch_shape, 9, 9, 6, device=device, dtype=dtype)
    dG[..., 0:3, 3:6] = dR
    dG[..., 3:6, 0:3] = dR
    dG[..., 3:6, 3:6] = (
        torch.matmul(Lie.so3hat(dv), R_for_directions)
        + torch.matmul(Lie.so3hat(v).unsqueeze(-3), dR)
    )
    dG[..., 6:9, 3:6] = (
        torch.matmul(Lie.so3hat(dp), R_for_directions)
        + torch.matmul(Lie.so3hat(p).unsqueeze(-3), dR)
    )
    return -dt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * dG


def preintegrate_interval_covariance_forward_sensitivity(
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    t: torch.Tensor,
    start_idx: int,
    end_idx: int,
    Qn: torch.Tensor,
    Qn_sensitivity: torch.Tensor | None = None,
    include_state_sensitivity: bool = False,
) -> IntervalCovarianceForwardSensitivity:
    """Propagate covariance and explicit first-order forward sensitivities.

    Noise-parameter directions obey the paper's fixed-trajectory recursion

    ``S_next = F S F^T + G (dQn) G^T``.

    A local left state direction obeys

    ``S_next = F S F^T + dG Qn G^T + G Qn dG^T``.

    There is deliberately no ``dF`` term: ``se23_phi`` depends on gravity and
    ``dt`` but not on the propagated state. ``se23_Gd`` depends on ``R,v,p``
    and is differentiated explicitly.
    """
    if end_idx <= start_idx:
        raise ValueError("end_idx must be greater than start_idx")
    if Qn.shape != (6, 6):
        raise ValueError(f"Qn must have shape (6, 6), got {tuple(Qn.shape)}")
    if Qn_sensitivity is not None:
        if Qn_sensitivity.ndim != 3 or Qn_sensitivity.shape[-2:] != (6, 6):
            raise ValueError(
                "Qn_sensitivity must have shape (num_directions, 6, 6)"
            )

    batch_shape = R.shape[1:-2]
    device = R.device
    dtype = R.dtype
    interval_steps = end_idx - start_idx
    P = torch.zeros(*batch_shape, 9, 9, device=device, dtype=dtype)
    Phi_total = _eye(9, batch_shape, device, dtype)
    Qn = Qn.to(device=device, dtype=dtype)

    noise_sensitivity = None
    Qn_sensitivity_view = None
    if Qn_sensitivity is not None:
        Qn_sensitivity = Qn_sensitivity.to(device=device, dtype=dtype)
        direction_count = Qn_sensitivity.shape[0]
        noise_sensitivity = torch.zeros(
            direction_count, *batch_shape, 9, 9, device=device, dtype=dtype
        )
        Qn_sensitivity_view = Qn_sensitivity.reshape(
            direction_count, *([1] * len(batch_shape)), 6, 6
        )

    state_sensitivity = None
    if include_state_sensitivity:
        state_sensitivity = torch.zeros(
            interval_steps, 9, *batch_shape, 9, 9, device=device, dtype=dtype
        )

    for local_step, step in enumerate(range(start_idx, end_idx)):
        dt = _step_dt(t, step).to(device=device, dtype=dtype)
        Phi_step = se23_phi(dt, device=device, dtype=dtype)
        Gd_step = se23_Gd(R[step], v[step], p[step], dt)
        Gd_transpose = Gd_step.transpose(-1, -2)

        P = _sandwich(Phi_step, P) + torch.matmul(
            torch.matmul(Gd_step, Qn), Gd_transpose
        )
        Phi_total = torch.matmul(Phi_step, Phi_total)

        if noise_sensitivity is not None:
            noise_sensitivity = _sandwich(Phi_step, noise_sensitivity)
            Gd_direction = Gd_step.unsqueeze(0)
            noise_sensitivity = noise_sensitivity + torch.matmul(
                torch.matmul(Gd_direction, Qn_sensitivity_view),
                Gd_direction.transpose(-1, -2),
            )

        if state_sensitivity is not None:
            state_sensitivity = _sandwich(Phi_step, state_sensitivity)
            dGd_step = se23_Gd_left_tangent_sensitivity(
                R[step], v[step], p[step], dt
            ).movedim(-3, 0)
            state_injection = torch.matmul(
                torch.matmul(dGd_step, Qn), Gd_step.transpose(-1, -2)
            ) + torch.matmul(
                torch.matmul(Gd_step, Qn), dGd_step.transpose(-1, -2)
            )
            state_sensitivity[local_step] = (
                state_sensitivity[local_step] + state_injection
            )

    if noise_sensitivity is not None:
        noise_sensitivity = symmetrize(noise_sensitivity)
    if state_sensitivity is not None:
        state_sensitivity = symmetrize(state_sensitivity)
    return IntervalCovarianceForwardSensitivity(
        Phi=Phi_total,
        covariance=symmetrize(P),
        noise=noise_sensitivity,
        state=state_sensitivity,
    )


@torch.no_grad()
def contract_interval_covariance_state_forward_sensitivity_chunked(
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    t: torch.Tensor,
    start_idx: int,
    end_idx: int,
    Qn: torch.Tensor,
    covariance_cotangent: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Propagate covariance-state sensitivities forward in bounded chunks.

    This is the memory-bounded implementation used inside the bias double
    adjoint. It is deliberately distinct from the forward sensitivities with
    respect to learned covariance/noise parameters.

    For every local state ``xi_j``, the method still evaluates the original
    forward recursion

    ``S_{k+1,j} = F_k S_{k,j} F_k^T + 1[k=j] d(G_k Qn G_k^T)/dxi_j``.

    State indices are processed in chunks. A chunk is propagated all the way
    to the interval end, contracted with ``dL/dQ_end``, and released before
    the next chunk. Only already-injected (active) sensitivities are advanced;
    future zero directions are never multiplied. No state or covariance value
    is detached by this optimization, and no reverse covariance adjoint is
    used.

    The result has shape ``[interval_steps, *batch, 9]`` and matches contraction
    of the fully materialized forward Jacobian with ``covariance_cotangent``.
    """
    if end_idx <= start_idx:
        raise ValueError("end_idx must be greater than start_idx")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if Qn.shape != (6, 6):
        raise ValueError(f"Qn must have shape (6, 6), got {tuple(Qn.shape)}")

    batch_shape = R.shape[1:-2]
    expected_cotangent_shape = (*batch_shape, 9, 9)
    if covariance_cotangent.shape != expected_cotangent_shape:
        raise ValueError(
            "covariance_cotangent must have shape "
            f"{expected_cotangent_shape}, got {tuple(covariance_cotangent.shape)}"
        )

    device = R.device
    dtype = R.dtype
    interval_steps = end_idx - start_idx
    Qn = Qn.to(device=device, dtype=dtype)
    # The primal covariance and the materialized sensitivity reference are
    # symmetrized at the interval end. Moving that linear operation to the
    # cotangent is exact: <C, sym(S)> = <sym(C), S>.
    covariance_cotangent = symmetrize(
        covariance_cotangent.to(device=device, dtype=dtype)
    )
    state_gradient = torch.empty(
        interval_steps,
        *batch_shape,
        9,
        device=device,
        dtype=dtype,
    )

    for chunk_start in range(0, interval_steps, chunk_size):
        chunk_end = min(chunk_start + chunk_size, interval_steps)
        chunk_length = chunk_end - chunk_start
        sensitivity = torch.zeros(
            chunk_length,
            9,
            *batch_shape,
            9,
            9,
            device=device,
            dtype=dtype,
        )

        for local_step in range(chunk_start, interval_steps):
            step = start_idx + local_step
            dt = _step_dt(t, step).to(device=device, dtype=dtype)
            Phi_step = se23_phi(dt, device=device, dtype=dtype)

            # At the start of step k, only sensitivities injected at j < k
            # are active and need the F_k S F_k^T propagation.
            active_count = min(max(local_step - chunk_start, 0), chunk_length)
            if active_count:
                propagated = _sandwich(Phi_step, sensitivity[:active_count])
                sensitivity[:active_count].copy_(propagated)
                del propagated

            if local_step < chunk_end:
                Gd_step = se23_Gd(R[step], v[step], p[step], dt)
                dGd_step = se23_Gd_left_tangent_sensitivity(
                    R[step], v[step], p[step], dt
                ).movedim(-3, 0)
                state_injection = torch.matmul(
                    torch.matmul(dGd_step, Qn),
                    Gd_step.transpose(-1, -2),
                ) + torch.matmul(
                    torch.matmul(Gd_step, Qn),
                    dGd_step.transpose(-1, -2),
                )
                sensitivity[local_step - chunk_start].copy_(state_injection)
                del Gd_step, dGd_step, state_injection

        state_gradient[chunk_start:chunk_end].copy_(
            torch.einsum(
                "ck...ij,...ij->c...k",
                sensitivity,
                covariance_cotangent,
            )
        )
        del sensitivity

    return state_gradient


@torch.no_grad()
def preintegrate_interval_covariance_state_vjp(
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    t: torch.Tensor,
    start_idx: int,
    end_idx: int,
    Qn: torch.Tensor,
    covariance_cotangent: torch.Tensor,
) -> torch.Tensor:
    """Contract an interval covariance Jacobian without materializing it.

    For the invariant covariance recursion

    ``P_{k+1} = F_k P_k F_k^T + G_k Qn G_k^T``,

    ``F_k`` is independent of the nominal state while ``G_k`` is not. Given
    the cotangent ``dL/dP_end``, this routine propagates

    ``Lambda_k = F_k^T Lambda_{k+1} F_k``

    backwards and contracts each local ``dG_k/dxi_k`` directly with the
    corresponding ``dL/dG_k``. The returned tensor has shape
    ``[interval_steps, *batch, 9]`` and is exactly the VJP that would be
    obtained by contracting ``covariance_cotangent`` with the explicit
    forward sensitivity ``dP_end/dxi_k``.

    The output covariance is symmetrized by the primal preintegrator, so the
    initial cotangent is symmetrized as the adjoint of that operation. No
    reverse-mode autograd graph is constructed.
    """
    if end_idx <= start_idx:
        raise ValueError("end_idx must be greater than start_idx")
    if Qn.shape != (6, 6):
        raise ValueError(f"Qn must have shape (6, 6), got {tuple(Qn.shape)}")

    batch_shape = R.shape[1:-2]
    expected_cotangent_shape = (*batch_shape, 9, 9)
    if covariance_cotangent.shape != expected_cotangent_shape:
        raise ValueError(
            "covariance_cotangent must have shape "
            f"{expected_cotangent_shape}, got {tuple(covariance_cotangent.shape)}"
        )

    device = R.device
    dtype = R.dtype
    Qn = Qn.to(device=device, dtype=dtype)
    Lambda = symmetrize(
        covariance_cotangent.to(device=device, dtype=dtype)
    )
    state_vjp = torch.empty(
        end_idx - start_idx,
        *batch_shape,
        9,
        device=device,
        dtype=dtype,
    )

    for step in range(end_idx - 1, start_idx - 1, -1):
        local_step = step - start_idx
        dt = _step_dt(t, step).to(device=device, dtype=dtype)
        Phi_step = se23_phi(dt, device=device, dtype=dtype)
        Gd_step = se23_Gd(R[step], v[step], p[step], dt)

        # For <Lambda, G Qn G^T>, the cotangent of G is
        # Lambda G Qn^T + Lambda^T G Qn. Keeping the unsimplified form makes
        # the contraction exact even if a caller supplies a non-symmetric Qn.
        Gd_cotangent = torch.matmul(
            torch.matmul(Lambda, Gd_step),
            Qn.transpose(-1, -2),
        ) + torch.matmul(
            torch.matmul(Lambda.transpose(-1, -2), Gd_step),
            Qn,
        )
        dGd_step = se23_Gd_left_tangent_sensitivity(
            R[step], v[step], p[step], dt
        )
        state_vjp[local_step] = torch.einsum(
            "...kij,...ij->...k",
            dGd_step,
            Gd_cotangent,
        )

        # Adjoint of P_{k+1} = F_k P_k F_k^T. There is deliberately no
        # state-gradient contribution from F_k because it is state-independent.
        Lambda = _sandwich(Phi_step.transpose(-1, -2), Lambda)

    return state_vjp


def preintegrate_supervision_covariances_forward_sensitivity(
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    t: torch.Tensor,
    supervised_indices: torch.Tensor,
    Qn: torch.Tensor,
    Qn_sensitivity: torch.Tensor | None = None,
    include_state_sensitivity: bool = False,
) -> SupervisionCovarianceForwardSensitivity:
    """Build nominal interval covariances and their forward sensitivities."""
    supervised_indices = supervised_indices.to(device=R.device, dtype=torch.long)
    if supervised_indices.dim() != 1 or supervised_indices.numel() < 2:
        raise ValueError(
            "supervised_indices must be a 1D tensor with at least two entries"
        )

    interval_results = []
    for i in range(supervised_indices.numel() - 1):
        interval_results.append(
            preintegrate_interval_covariance_forward_sensitivity(
                R,
                v,
                p,
                t,
                int(supervised_indices[i].item()),
                int(supervised_indices[i + 1].item()),
                Qn,
                Qn_sensitivity=Qn_sensitivity,
                include_state_sensitivity=include_state_sensitivity,
            )
        )

    noise_sensitivity = None
    if Qn_sensitivity is not None:
        noise_sensitivity = torch.stack(
            [result.noise for result in interval_results], dim=1
        )
    state_sensitivity = None
    if include_state_sensitivity:
        state_sensitivity = tuple(result.state for result in interval_results)
    return SupervisionCovarianceForwardSensitivity(
        Phi=torch.stack([result.Phi for result in interval_results], dim=0),
        covariance=torch.stack(
            [result.covariance for result in interval_results], dim=0
        ),
        noise=noise_sensitivity,
        state=state_sensitivity,
    )


def preintegrate_interval_covariance(
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    t: torch.Tensor,
    start_idx: int,
    end_idx: int,
    Qn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate the raw covariance for one supervision interval.

    No diagonal regularizer is added here.  Positive definiteness must come
    from the accumulated IMU noise injections over the supervision interval.
    """
    if end_idx <= start_idx:
        raise ValueError("end_idx must be greater than start_idx")

    batch_shape = R.shape[1:-2]
    device = R.device
    dtype = R.dtype
    P = torch.zeros(*batch_shape, 9, 9, device=device, dtype=dtype)
    Phi_total = _eye(9, batch_shape, device, dtype)
    Qn = Qn.to(device=device, dtype=dtype)

    for step in range(start_idx, end_idx):
        dt = _step_dt(t, step).to(device=device, dtype=dtype)
        Phi_step = se23_phi(dt, device=device, dtype=dtype)
        Gd_step = se23_Gd(R[step], v[step], p[step], dt)
        Qd_step = torch.matmul(torch.matmul(Gd_step, Qn), Gd_step.transpose(-1, -2))
        P = torch.matmul(torch.matmul(Phi_step, P), Phi_step.transpose(-1, -2)) + Qd_step
        Phi_total = torch.matmul(Phi_step, Phi_total)

    return Phi_total, symmetrize(P)


def preintegrate_supervision_covariances(
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    t: torch.Tensor,
    supervised_indices: torch.Tensor,
    Qn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build raw Phi_i^s and Q_i^s for all adjacent supervised states."""
    supervised_indices = supervised_indices.to(device=R.device, dtype=torch.long)
    if supervised_indices.dim() != 1 or supervised_indices.numel() < 2:
        raise ValueError("supervised_indices must be a 1D tensor with at least two entries")

    Phi_list = []
    Q_list = []
    for i in range(supervised_indices.numel() - 1):
        start_idx = int(supervised_indices[i].item())
        end_idx = int(supervised_indices[i + 1].item())
        Phi_i, Q_i = preintegrate_interval_covariance(
            R, v, p, t, start_idx, end_idx, Qn
        )
        Phi_list.append(Phi_i)
        Q_list.append(Q_i)
    return torch.stack(Phi_list, dim=0), torch.stack(Q_list, dim=0)


def build_se23_precision(
    Phi_list: torch.Tensor | list[torch.Tensor],
    Q_list: torch.Tensor | list[torch.Tensor],
    prior_std: float = 1e-4,
    jitter: float = 0.0,
) -> torch.Tensor:
    """Assemble dense storage for the block-tridiagonal SE_2(3) precision."""
    if isinstance(Phi_list, list):
        Phi_list = torch.stack(Phi_list, dim=0)
    if isinstance(Q_list, list):
        Q_list = torch.stack(Q_list, dim=0)
    if Phi_list.shape[0] != Q_list.shape[0]:
        raise ValueError("Phi_list and Q_list must have the same interval count")

    M = Phi_list.shape[0]
    dx = 9
    batch_shape = Phi_list.shape[1:-2]
    device = Phi_list.device
    dtype = Phi_list.dtype
    total_dim = (M + 1) * dx
    Lambda = torch.zeros(*batch_shape, total_dim, total_dim, device=device, dtype=dtype)
    prior_inv = (1.0 / (prior_std * prior_std)) * _eye(dx, batch_shape, device, dtype)
    Lambda[..., 0:dx, 0:dx] = Lambda[..., 0:dx, 0:dx] + prior_inv

    for i in range(M):
        Phi_i = Phi_list[i]
        Q_i = symmetrize(Q_list[i])
        Q_inv = cholesky_inverse_spd(Q_i, jitter=jitter)
        i0 = i * dx
        i1 = (i + 1) * dx
        block_ii = torch.matmul(torch.matmul(Phi_i.transpose(-1, -2), Q_inv), Phi_i)
        block_i_next = -torch.matmul(Phi_i.transpose(-1, -2), Q_inv)
        block_next_i = -torch.matmul(Q_inv, Phi_i)

        Lambda[..., i0:i0 + dx, i0:i0 + dx] = (
            Lambda[..., i0:i0 + dx, i0:i0 + dx] + block_ii
        )
        Lambda[..., i0:i0 + dx, i1:i1 + dx] = (
            Lambda[..., i0:i0 + dx, i1:i1 + dx] + block_i_next
        )
        Lambda[..., i1:i1 + dx, i0:i0 + dx] = (
            Lambda[..., i1:i1 + dx, i0:i0 + dx] + block_next_i
        )
        Lambda[..., i1:i1 + dx, i1:i1 + dx] = (
            Lambda[..., i1:i1 + dx, i1:i1 + dx] + Q_inv
        )

    return symmetrize(Lambda)


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _sandwich(left: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Return ``left @ matrix @ left.T`` with leading-direction broadcasting."""
    return torch.matmul(
        torch.matmul(left, matrix),
        left.transpose(-1, -2),
    )


def safe_cholesky(matrix: torch.Tensor, jitter: float = 0.0, max_tries: int = 5) -> torch.Tensor:
    """Factor an SPD matrix, with regularization only when explicitly requested.

    With ``jitter=0`` this is a strict factorization of the supplied matrix:
    no diagonal term is silently injected and failure is reported immediately.
    A positive jitter retains the opt-in geometric retry used by legacy runs.
    """
    matrix = symmetrize(matrix)
    if jitter < 0:
        raise ValueError("jitter must be non-negative")
    if jitter == 0:
        chol, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
        if torch.any(info != 0):
            failed = int(torch.count_nonzero(info).item())
            total = int(info.numel())
            raise RuntimeError(
                f"Cholesky failed for {failed}/{total} matrices with fixed jitter disabled"
            )
        return chol

    eye = _eye(matrix.shape[-1], matrix.shape[:-2], matrix.device, matrix.dtype)
    current_jitter = jitter
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(matrix + current_jitter * eye)
        except RuntimeError:
            current_jitter *= 10.0
    return torch.linalg.cholesky(matrix + current_jitter * eye)


def cholesky_inverse_spd(matrix: torch.Tensor, jitter: float = 0.0) -> torch.Tensor:
    chol = safe_cholesky(matrix, jitter=jitter)
    return torch.cholesky_inverse(chol)


def logdet_spd(matrix: torch.Tensor, jitter: float = 0.0) -> torch.Tensor:
    chol = safe_cholesky(matrix, jitter=jitter)
    return 2.0 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(dim=-1)


def _step_dt(t: torch.Tensor, step: int) -> torch.Tensor:
    return t[step + 1] - t[step]


def _as_tensor(
    value: torch.Tensor | float,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if device is None:
            device = value.device
        if dtype is None:
            dtype = value.dtype
        return value.to(device=device, dtype=dtype)
    if dtype is None:
        dtype = torch.get_default_dtype()
    return torch.tensor(value, device=device, dtype=dtype)


def _eye(
    n: int,
    batch_shape: torch.Size | tuple[int, ...],
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.eye(n, device=device, dtype=dtype).expand(*batch_shape, n, n)
