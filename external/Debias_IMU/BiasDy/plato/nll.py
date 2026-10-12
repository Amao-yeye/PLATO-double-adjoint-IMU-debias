from __future__ import annotations

import torch
from torch import nn

try:
    from plato.covariance import logdet_spd, safe_cholesky, symmetrize
except ImportError:
    from .covariance import logdet_spd, safe_cholesky, symmetrize


class PlatoNoiseParams(nn.Module):
    """Learnable IMU noise covariance outside the bias dynamics networks."""

    def __init__(
        self,
        init_sigma_a: float = 0.1,
        init_sigma_g: float = 0.1,
        isotropic: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.isotropic = isotropic
        shape = (1,) if isotropic else (3,)
        self.log_sigma_a = nn.Parameter(
            torch.full(shape, init_sigma_a, device=device, dtype=dtype).log()
        )
        self.log_sigma_g = nn.Parameter(
            torch.full(shape, init_sigma_g, device=device, dtype=dtype).log()
        )

    def sigmas(self) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_a = self.log_sigma_a.exp()
        sigma_g = self.log_sigma_g.exp()
        if self.isotropic:
            sigma_a = sigma_a.expand(3)
            sigma_g = sigma_g.expand(3)
        return sigma_a, sigma_g

    @torch.no_grad()
    def clamp_sigmas(
        self,
        min_sigma_a: float | None = None,
        max_sigma_a: float | None = None,
        min_sigma_g: float | None = None,
        max_sigma_g: float | None = None,
    ) -> None:
        """Clamp learnable noise standard deviations in log space."""
        if min_sigma_a is not None and min_sigma_a <= 0:
            raise ValueError("min_sigma_a must be positive")
        if min_sigma_g is not None and min_sigma_g <= 0:
            raise ValueError("min_sigma_g must be positive")
        if max_sigma_a is not None and max_sigma_a <= 0:
            raise ValueError("max_sigma_a must be positive")
        if max_sigma_g is not None and max_sigma_g <= 0:
            raise ValueError("max_sigma_g must be positive")
        if min_sigma_a is not None and max_sigma_a is not None and min_sigma_a > max_sigma_a:
            raise ValueError("min_sigma_a cannot exceed max_sigma_a")
        if min_sigma_g is not None and max_sigma_g is not None and min_sigma_g > max_sigma_g:
            raise ValueError("min_sigma_g cannot exceed max_sigma_g")

        min_log_a = None if min_sigma_a is None else torch.tensor(
            min_sigma_a, device=self.log_sigma_a.device, dtype=self.log_sigma_a.dtype
        ).log()
        max_log_a = None if max_sigma_a is None else torch.tensor(
            max_sigma_a, device=self.log_sigma_a.device, dtype=self.log_sigma_a.dtype
        ).log()
        min_log_g = None if min_sigma_g is None else torch.tensor(
            min_sigma_g, device=self.log_sigma_g.device, dtype=self.log_sigma_g.dtype
        ).log()
        max_log_g = None if max_sigma_g is None else torch.tensor(
            max_sigma_g, device=self.log_sigma_g.device, dtype=self.log_sigma_g.dtype
        ).log()

        if min_log_a is not None or max_log_a is not None:
            self.log_sigma_a.clamp_(min=min_log_a, max=max_log_a)
        if min_log_g is not None or max_log_g is not None:
            self.log_sigma_g.clamp_(min=min_log_g, max=max_log_g)

    def Qn(self) -> torch.Tensor:
        sigma_a, sigma_g = self.sigmas()
        variances = torch.cat([sigma_a.square(), sigma_g.square()], dim=0)
        return torch.diag(variances)

    def log_sigma_vector(self) -> torch.Tensor:
        """Return independent log-noise coordinates in sensitivity order.

        Isotropic noise has two coordinates ``[log_sigma_a, log_sigma_g]``;
        anisotropic noise has six, ordered as the three accelerometer axes
        followed by the three gyroscope axes.
        """
        return torch.cat(
            [self.log_sigma_a.reshape(-1), self.log_sigma_g.reshape(-1)], dim=0
        )

    def Qn_log_sigma_sensitivity(self) -> torch.Tensor:
        """Return ``dQn/d(log_sigma_j)`` for every independent coordinate.

        Since each learned coordinate is a log standard deviation, every
        active variance derivative is ``2 * sigma_j**2``. The returned tensor
        has shape ``[2, 6, 6]`` for isotropic noise and ``[6, 6, 6]`` for
        anisotropic noise, matching :meth:`log_sigma_vector`.
        """
        variance_a = self.log_sigma_a.exp().square()
        variance_g = self.log_sigma_g.exp().square()
        zeros3 = torch.zeros(3, device=variance_a.device, dtype=variance_a.dtype)

        if self.isotropic:
            diagonal_directions = torch.stack(
                [
                    torch.cat([2.0 * variance_a.expand(3), zeros3], dim=0),
                    torch.cat([zeros3, 2.0 * variance_g.expand(3)], dim=0),
                ],
                dim=0,
            )
        else:
            eye3 = torch.eye(3, device=variance_a.device, dtype=variance_a.dtype)
            accel_directions = torch.cat(
                [2.0 * variance_a.unsqueeze(-1) * eye3, torch.zeros_like(eye3)],
                dim=-1,
            )
            gyro_directions = torch.cat(
                [torch.zeros_like(eye3), 2.0 * variance_g.unsqueeze(-1) * eye3],
                dim=-1,
            )
            diagonal_directions = torch.cat(
                [accel_directions, gyro_directions], dim=0
            )
        return torch.diag_embed(diagonal_directions)


def _so3_right_jacobian_inverse_logabsdet_from_rotation(
    rotation_residual: torch.Tensor,
) -> torch.Tensor:
    """Return ``log|det J_r(phi)^{-1}|`` for an SO(3) residual."""
    if rotation_residual.shape[-1] != 3:
        raise ValueError("SO(3) residual must have last dimension 3")

    theta2 = rotation_residual.square().sum(dim=-1)
    small = theta2 < 1e-6

    # Keep the inactive exact branch finite at theta=0. torch.where evaluates
    # both branches, so a direct sqrt(theta2) would otherwise create a NaN VJP.
    theta_exact = torch.sqrt(torch.where(small, torch.ones_like(theta2), theta2))
    half_theta = 0.5 * theta_exact
    exact = -2.0 * (
        torch.log(torch.abs(torch.sin(half_theta))) - torch.log(half_theta)
    )

    theta4 = theta2.square()
    series = theta2 / 12.0 + theta4 / 1440.0 + theta4 * theta2 / 90720.0
    return torch.where(small, series, exact)


def so3_right_jacobian_inverse_logabsdet(
    residual: torch.Tensor,
) -> torch.Tensor:
    """Return the right-Jacobian coordinate term for rotation-only NLL."""
    return _so3_right_jacobian_inverse_logabsdet_from_rotation(residual)


def se23_right_jacobian_inverse_logabsdet(
    residual: torch.Tensor,
) -> torch.Tensor:
    """Return the paper's per-state ``log|det H|`` for an SE_2(3) residual.

    For the right-invariant residual ``r = Log(X_gt X_pred^{-1})``, Eq. (9)
    uses ``H = -dexp^{-1}_{-r} = -J_r(r)^{-1}``.  The SE_2(3) Jacobian is
    block lower triangular with three identical SO(3) Jacobians on its
    diagonal.  Consequently, with ``theta = ||r_rot||``,

        log|det H| = -6 log(sin(theta / 2) / (theta / 2)).

    A sixth-order series is used near zero so that both the value and the
    autograd derivative remain finite at an exactly zero rotation residual.
    Translation and velocity residuals do not affect the determinant.
    """
    if residual.shape[-1] != 9:
        raise ValueError("SE_2(3) residual must have last dimension 9")

    # SE_2(3) has three identical SO(3) Jacobian blocks on its diagonal.
    return 3.0 * _so3_right_jacobian_inverse_logabsdet_from_rotation(
        residual[..., :3]
    )


def full_state_nll(
    residual: torch.Tensor,
    Lambda: torch.Tensor,
    meas_sigma_rot: float = 1e-3,
    meas_sigma_vel: float = 1e-2,
    meas_sigma_pos: float = 1e-2,
    jitter: float = 0.0,
    reduction: str = "mean",
    time_first: bool | None = None,
    include_logdet: bool = True,
) -> torch.Tensor:
    """NLL for full SE_2(3) residuals [phi, rho_v, rho_p]."""
    device = residual.device
    dtype = residual.dtype
    H_block = torch.eye(9, device=device, dtype=dtype)
    W_block = torch.diag(torch.tensor(
        [meas_sigma_rot ** 2] * 3
        + [meas_sigma_vel ** 2] * 3
        + [meas_sigma_pos ** 2] * 3,
        device=device,
        dtype=dtype,
    ))
    return woodbury_nll(
        residual,
        Lambda,
        H_block,
        W_block,
        jitter=jitter,
        reduction=reduction,
        time_first=time_first,
        include_logdet=include_logdet,
    )


def full_state_nees(
    residual: torch.Tensor,
    Lambda: torch.Tensor,
    meas_sigma_rot: float = 1e-3,
    meas_sigma_vel: float = 1e-2,
    meas_sigma_pos: float = 1e-2,
    jitter: float = 0.0,
    reduction: str = "mean",
    time_first: bool | None = None,
) -> torch.Tensor:
    """NEES for full SE_2(3) residuals under the Woodbury marginal covariance."""
    return 2.0 * full_state_nll(
        residual,
        Lambda,
        meas_sigma_rot=meas_sigma_rot,
        meas_sigma_vel=meas_sigma_vel,
        meas_sigma_pos=meas_sigma_pos,
        jitter=jitter,
        reduction=reduction,
        time_first=time_first,
        include_logdet=False,
    )


def pose_nll(
    residual: torch.Tensor,
    Lambda: torch.Tensor,
    meas_sigma_rot: float = 1e-2,
    meas_sigma_pos: float = 5e-2,
    jitter: float = 0.0,
    reduction: str = "mean",
    time_first: bool | None = None,
    include_logdet: bool = True,
) -> torch.Tensor:
    """NLL for pose residuals [phi, rho_p]."""
    device = residual.device
    dtype = residual.dtype
    H_block = torch.zeros(6, 9, device=device, dtype=dtype)
    H_block[0:3, 0:3] = torch.eye(3, device=device, dtype=dtype)
    H_block[3:6, 6:9] = torch.eye(3, device=device, dtype=dtype)
    W_block = torch.diag(torch.tensor(
        [meas_sigma_rot ** 2] * 3 + [meas_sigma_pos ** 2] * 3,
        device=device,
        dtype=dtype,
    ))
    return woodbury_nll(
        residual,
        Lambda,
        H_block,
        W_block,
        jitter=jitter,
        reduction=reduction,
        time_first=time_first,
        include_logdet=include_logdet,
    )


def pose_nees(
    residual: torch.Tensor,
    Lambda: torch.Tensor,
    meas_sigma_rot: float = 1e-2,
    meas_sigma_pos: float = 5e-2,
    jitter: float = 0.0,
    reduction: str = "mean",
    time_first: bool | None = None,
) -> torch.Tensor:
    """NEES for pose residuals [phi, rho_p] under the Woodbury marginal covariance."""
    return 2.0 * pose_nll(
        residual,
        Lambda,
        meas_sigma_rot=meas_sigma_rot,
        meas_sigma_pos=meas_sigma_pos,
        jitter=jitter,
        reduction=reduction,
        time_first=time_first,
        include_logdet=False,
    )


def sparse_pose_nll(
    residual: torch.Tensor,
    Phi_list: torch.Tensor | list[torch.Tensor],
    Q_list: torch.Tensor | list[torch.Tensor],
    meas_sigma_rot: float = 1e-2,
    meas_sigma_pos: float = 5e-2,
    prior_std: float = 1e-4,
    P0: torch.Tensor | None = None,
    include_logdet: bool = True,
    jitter: float = 0.0,
    reduction: str = "mean",
    time_first: bool | None = None,
) -> torch.Tensor:
    """Sparse noisy SE(3) pose NLL with SE_2(3) Markov transitions.

    The pose residual order is [delta_theta, delta_p].  The latent state order
    is [delta_theta, delta_v, delta_p].  This implements the block-tridiagonal
    Woodbury form from docs/sparse_noisy_vo_nll_for_codex.md without forming the
    dense marginal pose covariance.
    """
    Phi_list, Q_list = _normalize_transition_blocks(Phi_list, Q_list)
    residual = _normalize_pose_residual(
        residual,
        num_states=Phi_list.shape[0] + 1,
        time_first=time_first,
    )

    Phi = Phi_list.transpose(0, 1)
    Q = Q_list.transpose(0, 1)
    Bsz, num_states, _ = residual.shape
    num_intervals = num_states - 1
    device = residual.device
    dtype = residual.dtype

    I9 = torch.eye(9, device=device, dtype=dtype)
    H = torch.zeros(6, 9, device=device, dtype=dtype)
    H[0:3, 0:3] = torch.eye(3, device=device, dtype=dtype)
    H[3:6, 6:9] = torch.eye(3, device=device, dtype=dtype)
    W = torch.diag(torch.tensor(
        [meas_sigma_rot ** 2] * 3 + [meas_sigma_pos ** 2] * 3,
        device=device,
        dtype=dtype,
    ))
    chol_W = safe_cholesky(W, jitter=jitter)
    W_inv_r = torch.cholesky_solve(
        residual.reshape(-1, 6, 1),
        chol_W.expand(Bsz * num_states, -1, -1),
    ).reshape(Bsz, num_states, 6)
    W_inv_H = torch.cholesky_solve(H, chol_W)

    residual_term = (residual * W_inv_r).sum(dim=(-1, -2))
    gamma = torch.matmul(W_inv_r, H)
    Omega = symmetrize(torch.matmul(H.transpose(-1, -2), W_inv_H))
    logdet_W = logdet_spd(W, jitter=jitter) * num_states

    Q = symmetrize(Q)
    chol_Q = safe_cholesky(Q, jitter=jitter)
    eye9_q = I9.expand(Bsz, num_intervals, 9, 9)
    Q_inv = torch.cholesky_solve(
        eye9_q.reshape(Bsz * num_intervals, 9, 9),
        chol_Q.reshape(Bsz * num_intervals, 9, 9),
    ).reshape(Bsz, num_intervals, 9, 9)

    if P0 is None:
        prior_var = torch.as_tensor(prior_std * prior_std, device=device, dtype=dtype)
        P0 = prior_var * I9
    P0 = P0.to(device=device, dtype=dtype)
    if P0.dim() == 2:
        P0 = P0.expand(Bsz, -1, -1)
    elif P0.dim() == 3 and P0.shape[0] == 1 and Bsz != 1:
        P0 = P0.expand(Bsz, -1, -1)
    elif P0.dim() != 3 or P0.shape[0] != Bsz:
        raise ValueError("P0 must have shape [9, 9], [1, 9, 9], or [B, 9, 9]")
    P0 = symmetrize(P0)
    chol_P0 = safe_cholesky(P0, jitter=jitter)
    P0_inv = torch.cholesky_solve(I9.expand(Bsz, 9, 9), chol_P0)

    A_blocks = []
    B_blocks = []
    omega = Omega.expand(Bsz, -1, -1)
    A0 = (
        P0_inv
        + torch.matmul(torch.matmul(Phi[:, 0].transpose(-1, -2), Q_inv[:, 0]), Phi[:, 0])
        + omega
    )
    A_blocks.append(symmetrize(A0))
    for i in range(1, num_intervals):
        Ai = (
            Q_inv[:, i - 1]
            + torch.matmul(torch.matmul(Phi[:, i].transpose(-1, -2), Q_inv[:, i]), Phi[:, i])
            + omega
        )
        A_blocks.append(symmetrize(Ai))
    AM = Q_inv[:, num_intervals - 1] + omega
    A_blocks.append(symmetrize(AM))

    for i in range(num_intervals):
        B_blocks.append(-torch.matmul(Phi[:, i].transpose(-1, -2), Q_inv[:, i]))

    D_chol = []
    chol_D0 = safe_cholesky(A_blocks[0], jitter=jitter)
    D_chol.append(chol_D0)
    logdet_K = 2.0 * torch.diagonal(chol_D0, dim1=-2, dim2=-1).log().sum(dim=-1)
    for i in range(1, num_states):
        X = torch.cholesky_solve(B_blocks[i - 1], D_chol[i - 1])
        Di = A_blocks[i] - torch.matmul(B_blocks[i - 1].transpose(-1, -2), X)
        chol_Di = safe_cholesky(Di, jitter=jitter)
        D_chol.append(chol_Di)
        logdet_K = logdet_K + 2.0 * torch.diagonal(chol_Di, dim1=-2, dim2=-1).log().sum(dim=-1)

    y = [gamma[:, 0]]
    for i in range(1, num_states):
        prev = torch.cholesky_solve(y[i - 1].unsqueeze(-1), D_chol[i - 1]).squeeze(-1)
        y.append(gamma[:, i] - torch.matmul(B_blocks[i - 1].transpose(-1, -2), prev.unsqueeze(-1)).squeeze(-1))

    u = [
        torch.cholesky_solve(y_i.unsqueeze(-1), D_i).squeeze(-1)
        for y_i, D_i in zip(y, D_chol)
    ]

    z = [None] * num_states
    z[-1] = u[-1]
    for i in range(num_states - 2, -1, -1):
        rhs = torch.matmul(B_blocks[i], z[i + 1].unsqueeze(-1)).squeeze(-1)
        correction = torch.cholesky_solve(rhs.unsqueeze(-1), D_chol[i]).squeeze(-1)
        z[i] = u[i] - correction
    z_tensor = torch.stack(z, dim=1)
    gamma_z = (gamma * z_tensor).sum(dim=(-1, -2))

    nll = 0.5 * residual_term - 0.5 * gamma_z
    if include_logdet:
        logdet_P0 = 2.0 * torch.diagonal(chol_P0, dim1=-2, dim2=-1).log().sum(dim=-1)
        logdet_Q = 2.0 * torch.diagonal(chol_Q, dim1=-2, dim2=-1).log().sum(dim=(-1, -2))
        nll = nll + 0.5 * logdet_W + 0.5 * logdet_K + 0.5 * logdet_P0 + 0.5 * logdet_Q

    if reduction == "none":
        return nll
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    raise ValueError("reduction must be one of: none, mean, sum")


def sparse_pose_nees(
    residual: torch.Tensor,
    Phi_list: torch.Tensor | list[torch.Tensor],
    Q_list: torch.Tensor | list[torch.Tensor],
    meas_sigma_rot: float = 1e-2,
    meas_sigma_pos: float = 5e-2,
    prior_std: float = 1e-4,
    P0: torch.Tensor | None = None,
    jitter: float = 0.0,
    reduction: str = "mean",
    time_first: bool | None = None,
) -> torch.Tensor:
    """NEES for sparse noisy SE(3) pose residuals."""
    return 2.0 * sparse_pose_nll(
        residual,
        Phi_list,
        Q_list,
        meas_sigma_rot=meas_sigma_rot,
        meas_sigma_pos=meas_sigma_pos,
        prior_std=prior_std,
        P0=P0,
        include_logdet=False,
        jitter=jitter,
        reduction=reduction,
        time_first=time_first,
    )


def sparse_rotation_nll(
    residual: torch.Tensor,
    Phi_list: torch.Tensor | list[torch.Tensor],
    Q_list: torch.Tensor | list[torch.Tensor],
    prior_std: float = 1e-4,
    include_prior: bool = False,
    include_logdet: bool = True,
    include_logdet_h: bool = True,
    jitter: float = 0.0,
    reduction: str = "mean",
    time_first: bool | None = None,
) -> torch.Tensor:
    """Rotation marginal of the sparse SE_2(3) transition-factor NLL.

    Rotation is an autonomous leading subsystem of the invariant SE_2(3)
    error dynamics.  Selecting ``Phi[:3, :3]`` and ``Q[:3, :3]`` therefore
    gives its exact Gaussian marginal, while retaining the temporal
    correlation and the SO(3) residual-coordinate normalizer.
    """
    Phi_list, Q_list = _normalize_transition_blocks(Phi_list, Q_list)
    residual = _normalize_transition_residual(
        residual,
        num_states=Phi_list.shape[0] + 1,
        time_first=time_first,
    )

    Phi = Phi_list[..., :3, :3].transpose(0, 1)
    Q = symmetrize(Q_list[..., :3, :3].transpose(0, 1))
    rotation_residual = residual[..., :3]
    logdet_h = so3_right_jacobian_inverse_logabsdet(rotation_residual).sum(dim=-1)
    rotation_residual = -rotation_residual

    pred_next = torch.matmul(
        Phi, rotation_residual[:, :-1].unsqueeze(-1)
    ).squeeze(-1)
    transition_residual = rotation_residual[:, 1:] - pred_next

    chol_Q = safe_cholesky(Q, jitter=jitter)
    solved = torch.cholesky_solve(
        transition_residual.unsqueeze(-1), chol_Q
    ).squeeze(-1)
    transition_quad = (transition_residual * solved).sum(dim=(-1, -2))

    nll = 0.5 * transition_quad
    if include_logdet:
        logdet_Q = 2.0 * torch.diagonal(
            chol_Q, dim1=-2, dim2=-1
        ).log().sum(dim=(-1, -2))
        nll = nll + 0.5 * logdet_Q
    if include_logdet_h:
        nll = nll + logdet_h

    if include_prior:
        prior_var = prior_std * prior_std
        prior_quad = rotation_residual[:, 0].square().sum(dim=-1) / prior_var
        nll = nll + 0.5 * prior_quad
        if include_logdet:
            logdet_prior = 3.0 * torch.as_tensor(
                prior_var, device=residual.device, dtype=residual.dtype
            ).log()
            nll = nll + 0.5 * logdet_prior

    if reduction == "none":
        return nll
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    raise ValueError("reduction must be one of: none, mean, sum")


def sparse_rotation_nees(
    residual: torch.Tensor,
    Phi_list: torch.Tensor | list[torch.Tensor],
    Q_list: torch.Tensor | list[torch.Tensor],
    prior_std: float = 1e-4,
    include_prior: bool = False,
    jitter: float = 0.0,
    reduction: str = "mean",
    time_first: bool | None = None,
) -> torch.Tensor:
    """NEES for the sparse rotation marginal."""
    return 2.0 * sparse_rotation_nll(
        residual,
        Phi_list,
        Q_list,
        prior_std=prior_std,
        include_prior=include_prior,
        include_logdet=False,
        include_logdet_h=False,
        jitter=jitter,
        reduction=reduction,
        time_first=time_first,
    )


def sparse_full_state_nll(
    residual: torch.Tensor,
    Phi_list: torch.Tensor | list[torch.Tensor],
    Q_list: torch.Tensor | list[torch.Tensor],
    prior_std: float = 1e-4,
    include_prior: bool = False,
    include_logdet: bool = True,
    include_logdet_h: bool = True,
    jitter: float = 0.0,
    reduction: str = "mean",
    time_first: bool | None = None,
) -> torch.Tensor:
    """Sparse transition-factor NLL for full SE_2(3) GT residuals.

    The residual order is [delta_theta, delta_v, delta_p].  For the
    right-invariant full-state GT residual, the transformed residual is -r.
    The global sign cancels in the transition quadratic, but keeping it here
    matches the zero-noise derivation. ``include_logdet`` controls the Gaussian
    covariance normalizer ``0.5 log det Q``; ``include_logdet_h`` independently
    controls the paper's residual-coordinate term ``sum_i log|det H_i|``.
    """
    Phi_list, Q_list = _normalize_transition_blocks(Phi_list, Q_list)
    residual = _normalize_transition_residual(
        residual,
        num_states=Phi_list.shape[0] + 1,
        time_first=time_first,
    )

    Phi = Phi_list.transpose(0, 1)
    Q = Q_list.transpose(0, 1)
    logdet_h = se23_right_jacobian_inverse_logabsdet(residual).sum(dim=-1)
    residual = -residual

    Q = symmetrize(Q)

    pred_next = torch.matmul(Phi, residual[:, :-1].unsqueeze(-1)).squeeze(-1)
    transition_residual = residual[:, 1:] - pred_next

    chol_Q = safe_cholesky(Q, jitter=jitter)
    solved = torch.cholesky_solve(transition_residual.unsqueeze(-1), chol_Q).squeeze(-1)
    transition_quad = (transition_residual * solved).sum(dim=(-1, -2))

    nll = 0.5 * transition_quad
    if include_logdet:
        logdet_Q = 2.0 * torch.diagonal(chol_Q, dim1=-2, dim2=-1).log().sum(dim=(-1, -2))
        nll = nll + 0.5 * logdet_Q
    if include_logdet_h:
        nll = nll + logdet_h

    if include_prior:
        prior_var = prior_std * prior_std
        prior_quad = residual[:, 0].square().sum(dim=-1) / prior_var
        nll = nll + 0.5 * prior_quad
        if include_logdet:
            logdet_prior = 9.0 * torch.as_tensor(
                prior_var, device=residual.device, dtype=residual.dtype
            ).log()
            nll = nll + 0.5 * logdet_prior

    if reduction == "none":
        return nll
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    raise ValueError("reduction must be one of: none, mean, sum")


def sparse_full_state_nees(
    residual: torch.Tensor,
    Phi_list: torch.Tensor | list[torch.Tensor],
    Q_list: torch.Tensor | list[torch.Tensor],
    prior_std: float = 1e-4,
    include_prior: bool = False,
    jitter: float = 0.0,
    reduction: str = "mean",
    time_first: bool | None = None,
) -> torch.Tensor:
    """NEES for sparse full-state SE_2(3) transition residuals."""
    return 2.0 * sparse_full_state_nll(
        residual,
        Phi_list,
        Q_list,
        prior_std=prior_std,
        include_prior=include_prior,
        include_logdet=False,
        include_logdet_h=False,
        jitter=jitter,
        reduction=reduction,
        time_first=time_first,
    )


def woodbury_nll(
    residual: torch.Tensor,
    Lambda: torch.Tensor,
    H_block: torch.Tensor,
    W_block: torch.Tensor,
    jitter: float = 0.0,
    reduction: str = "mean",
    time_first: bool | None = None,
    include_logdet: bool = True,
) -> torch.Tensor:
    """Compute Gaussian-process marginal NLL using the Woodbury precision form."""
    residual, Lambda = _normalize_residual_and_precision(
        residual, Lambda, H_block.shape[-1], time_first=time_first
    )
    B, num_states, meas_dim = residual.shape
    state_dim = H_block.shape[-1]
    if H_block.shape[0] != meas_dim:
        raise ValueError("H_block row count must match residual measurement dimension")

    device = residual.device
    dtype = residual.dtype
    H_block = H_block.to(device=device, dtype=dtype)
    W_block = W_block.to(device=device, dtype=dtype)
    W_inv = _inverse_spd(W_block, jitter=jitter)

    weighted_r = torch.matmul(residual, W_inv.transpose(-1, -2))
    r_quad = (weighted_r * residual).sum(dim=(-1, -2))
    eta = torch.matmul(weighted_r, H_block).reshape(B, num_states * state_dim)

    obs_precision_block = torch.matmul(torch.matmul(H_block.transpose(-1, -2), W_inv), H_block)
    obs_precision = torch.zeros(
        B, num_states * state_dim, num_states * state_dim, device=device, dtype=dtype
    )
    for i in range(num_states):
        s = i * state_dim
        obs_precision[:, s:s + state_dim, s:s + state_dim] = obs_precision_block

    K = symmetrize(Lambda + obs_precision)
    chol_K = safe_cholesky(K, jitter=jitter)
    solve_K_eta = torch.cholesky_solve(eta.unsqueeze(-1), chol_K).squeeze(-1)
    nll = 0.5 * r_quad - 0.5 * (eta * solve_K_eta).sum(dim=-1)
    if include_logdet:
        logdet_W = logdet_spd(W_block, jitter=jitter) * num_states
        logdet_K = 2.0 * torch.diagonal(chol_K, dim1=-2, dim2=-1).log().sum(dim=-1)
        logdet_Lambda = logdet_spd(Lambda, jitter=jitter)
        nll = nll + 0.5 * logdet_W + 0.5 * logdet_K - 0.5 * logdet_Lambda
    if reduction == "none":
        return nll
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    raise ValueError("reduction must be one of: none, mean, sum")


def woodbury_nees(
    residual: torch.Tensor,
    Lambda: torch.Tensor,
    H_block: torch.Tensor,
    W_block: torch.Tensor,
    jitter: float = 0.0,
    reduction: str = "mean",
    time_first: bool | None = None,
) -> torch.Tensor:
    """Compute NEES using the Woodbury marginal precision."""
    return 2.0 * woodbury_nll(
        residual,
        Lambda,
        H_block,
        W_block,
        jitter=jitter,
        reduction=reduction,
        time_first=time_first,
        include_logdet=False,
    )


def _inverse_spd(matrix: torch.Tensor, jitter: float) -> torch.Tensor:
    chol = safe_cholesky(matrix, jitter=jitter)
    return torch.cholesky_inverse(chol)


def _normalize_transition_blocks(
    Phi_list: torch.Tensor | list[torch.Tensor],
    Q_list: torch.Tensor | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(Phi_list, list):
        Phi_list = torch.stack(Phi_list, dim=0)
    if isinstance(Q_list, list):
        Q_list = torch.stack(Q_list, dim=0)
    if Phi_list.shape != Q_list.shape:
        raise ValueError("Phi_list and Q_list must have matching shapes")
    if Phi_list.dim() < 3 or Phi_list.shape[-2:] != (9, 9):
        raise ValueError("Phi_list and Q_list must have shape [M, ..., 9, 9]")
    if Phi_list.shape[0] < 1:
        raise ValueError("At least one transition block is required")
    if Phi_list.dim() == 3:
        Phi_list = Phi_list.unsqueeze(1)
        Q_list = Q_list.unsqueeze(1)
    return Phi_list, Q_list


def _normalize_transition_residual(
    residual: torch.Tensor,
    num_states: int,
    time_first: bool | None,
) -> torch.Tensor:
    if residual.dim() == 2:
        residual = residual.unsqueeze(1)
    elif residual.dim() != 3:
        raise ValueError("residual must have shape [T, B, 9], [B, T, 9], or [T, 9]")
    if residual.shape[-1] != 9:
        raise ValueError(f"Expected residual last dim 9, got {residual.shape[-1]}")

    if time_first is True:
        if residual.shape[0] != num_states:
            raise ValueError("time_first=True but residual first dimension is not num_states")
        residual = residual.transpose(0, 1)
    elif time_first is False:
        if residual.shape[1] != num_states:
            raise ValueError("time_first=False but residual second dimension is not num_states")
    elif residual.shape[0] == num_states and residual.shape[1] != num_states:
        residual = residual.transpose(0, 1)
    elif residual.shape[1] == num_states and residual.shape[0] != num_states:
        pass
    elif residual.shape[0] == num_states and residual.shape[1] == num_states:
        residual = residual.transpose(0, 1)
    else:
        raise ValueError("residual state count does not match transition blocks")
    return residual


def _normalize_pose_residual(
    residual: torch.Tensor,
    num_states: int,
    time_first: bool | None,
) -> torch.Tensor:
    if residual.dim() == 2:
        residual = residual.unsqueeze(1)
    elif residual.dim() != 3:
        raise ValueError("residual must have shape [T, B, 6], [B, T, 6], or [T, 6]")
    if residual.shape[-1] != 6:
        raise ValueError(f"Expected pose residual last dim 6, got {residual.shape[-1]}")

    if time_first is True:
        if residual.shape[0] != num_states:
            raise ValueError("time_first=True but residual first dimension is not num_states")
        residual = residual.transpose(0, 1)
    elif time_first is False:
        if residual.shape[1] != num_states:
            raise ValueError("time_first=False but residual second dimension is not num_states")
    elif residual.shape[0] == num_states and residual.shape[1] != num_states:
        residual = residual.transpose(0, 1)
    elif residual.shape[1] == num_states and residual.shape[0] != num_states:
        pass
    elif residual.shape[0] == num_states and residual.shape[1] == num_states:
        residual = residual.transpose(0, 1)
    else:
        raise ValueError("residual state count does not match transition blocks")
    return residual


def _normalize_residual_and_precision(
    residual: torch.Tensor,
    Lambda: torch.Tensor,
    state_dim: int,
    time_first: bool | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if Lambda.dim() == 2:
        Lambda = Lambda.unsqueeze(0)
    if Lambda.dim() != 3:
        raise ValueError("Lambda must have shape [B, N, N] or [N, N]")
    if Lambda.shape[-1] != Lambda.shape[-2] or Lambda.shape[-1] % state_dim != 0:
        raise ValueError("Lambda shape is incompatible with state_dim")
    num_states = Lambda.shape[-1] // state_dim

    if residual.dim() == 2:
        residual = residual.unsqueeze(1)
    elif residual.dim() != 3:
        raise ValueError("residual must have shape [T, B, D], [B, T, D], or [T, D]")

    if time_first is True:
        if residual.shape[0] != num_states:
            raise ValueError("time_first=True but residual first dimension does not match Lambda")
        residual = residual.transpose(0, 1)
    elif time_first is False:
        if residual.shape[1] != num_states:
            raise ValueError("time_first=False but residual second dimension does not match Lambda")
    elif residual.shape[0] == num_states and residual.shape[1] != num_states:
        residual = residual.transpose(0, 1)
    elif residual.shape[1] == num_states and residual.shape[0] != num_states:
        pass
    elif residual.shape[0] == num_states and residual.shape[1] == num_states:
        residual = residual.transpose(0, 1)
    else:
        raise ValueError("residual state count does not match Lambda")

    if Lambda.shape[0] == 1 and residual.shape[0] != 1:
        Lambda = Lambda.expand(residual.shape[0], -1, -1)
    elif residual.shape[0] != Lambda.shape[0]:
        raise ValueError("residual batch size does not match Lambda batch size")

    return residual, Lambda
