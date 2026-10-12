from __future__ import annotations

import torch

try:
    import lie_algebra as Lie
except ImportError:  # Support `python -m BiasDy.plato...` from Debias_IMU root.
    from BiasDy import lie_algebra as Lie


SE23_BLOCKS = {
    "rot": slice(0, 3),
    "vel": slice(3, 6),
    "pos": slice(6, 9),
}

POSE_BLOCKS = {
    "rot": slice(0, 3),
    "pos": slice(3, 6),
}


def make_se23_from_solution(R_pred: torch.Tensor, solution: torch.Tensor) -> torch.Tensor:
    """Build SE_2(3) matrices from ODE output.

    The ODE state order is [xi, t, bw, v, p, ba].
    """
    v_pred = solution[..., 7:10]
    p_pred = solution[..., 10:13]
    return Lie.SEn3fromSO3Rn(R_pred, torch.cat([v_pred, p_pred], dim=-1))


def se23_log_residual(
    X_ref: torch.Tensor,
    R_pred: torch.Tensor,
    solution: torch.Tensor,
) -> torch.Tensor:
    """Return log(X_ref^{-1} X_pred) with order [phi, rho_v, rho_p]."""
    X_pred = make_se23_from_solution(R_pred, solution)
    X_err = torch.matmul(Lie.SEn3inverse(X_ref), X_pred)
    return Lie.SEn3log(X_err)


def se23_gt_right_residual(
    X_ref: torch.Tensor,
    R_pred: torch.Tensor,
    solution: torch.Tensor,
) -> torch.Tensor:
    """Return Log(X_ref X_pred^{-1}) with order [phi, rho_v, rho_p]."""
    R_ref = X_ref[..., :3, :3]
    v_ref = X_ref[..., :3, 3]
    p_ref = X_ref[..., :3, 4]
    v_pred = solution[..., 7:10]
    p_pred = solution[..., 10:13]
    R_err = torch.matmul(R_ref, R_pred.transpose(-1, -2))
    v_err = v_ref - torch.matmul(R_err, v_pred.unsqueeze(-1)).squeeze(-1)
    p_err = p_ref - torch.matmul(R_err, p_pred.unsqueeze(-1)).squeeze(-1)
    X_err = Lie.SEn3fromSO3Rn(R_err, torch.cat((v_err, p_err), dim=-1))
    return Lie.SEn3log(X_err)


def se23_global_residual(
    X_ref: torch.Tensor,
    R_pred: torch.Tensor,
    solution: torch.Tensor,
) -> torch.Tensor:
    """Return direct state residual with order [phi, dv, dp].

    This keeps the SE_2(3) block structure but aligns the translational
    residuals with the global velocity/position RMSE used at evaluation.
    """
    R_ref = X_ref[..., :3, :3]
    v_ref = X_ref[..., :3, 3]
    p_ref = X_ref[..., :3, 4]
    phi = Lie.SO3log(torch.matmul(R_pred.transpose(-1, -2), R_ref))
    dv = solution[..., 7:10] - v_ref
    dp = solution[..., 10:13] - p_ref
    return torch.cat((phi, dv, dp), dim=-1)


def se23_pose_residual(
    R_ref: torch.Tensor,
    p_ref: torch.Tensor,
    R_pred: torch.Tensor,
    p_pred: torch.Tensor,
) -> torch.Tensor:
    """Return pose residual log(T_ref^{-1} T_pred) with order [phi, rho_p]."""
    R_err = torch.matmul(R_ref.transpose(-1, -2), R_pred)
    phi = Lie.SO3log(R_err)
    p_err = torch.matmul(
        R_ref.transpose(-1, -2),
        (p_pred - p_ref).unsqueeze(-1),
    ).squeeze(-1)
    rho_p = torch.matmul(
        Lie.SO3leftJacoInv(phi),
        p_err.unsqueeze(-1),
    ).squeeze(-1)
    return torch.cat([phi, rho_p], dim=-1)


def pose_right_residual(
    R_ref: torch.Tensor,
    p_ref: torch.Tensor,
    R_pred: torch.Tensor,
    p_pred: torch.Tensor,
) -> torch.Tensor:
    """Return pose residual log(T_ref T_pred^{-1}) with order [phi, rho_p]."""
    R_err = torch.matmul(R_ref, R_pred.transpose(-1, -2))
    phi = Lie.SO3log(R_err)
    p_err = p_ref - torch.matmul(R_err, p_pred.unsqueeze(-1)).squeeze(-1)
    rho_p = torch.matmul(
        Lie.SO3leftJacoInv(phi),
        p_err.unsqueeze(-1),
    ).squeeze(-1)
    return torch.cat([phi, rho_p], dim=-1)


def pose_global_residual(
    R_ref: torch.Tensor,
    p_ref: torch.Tensor,
    R_pred: torch.Tensor,
    p_pred: torch.Tensor,
) -> torch.Tensor:
    """Return direct pose residual with order [phi, dp]."""
    phi = Lie.SO3log(torch.matmul(R_pred.transpose(-1, -2), R_ref))
    dp = p_pred - p_ref
    return torch.cat([phi, dp], dim=-1)


def relative_pose_residual(
    R_ref: torch.Tensor,
    p_ref: torch.Tensor,
    R_pred: torch.Tensor,
    p_pred: torch.Tensor,
    mode: str = "log",
) -> torch.Tensor:
    """Return residuals of pose increments relative to the first timestamp.

    This is intended for low-rate VO supervision where the absolute VO pose may
    have a per-window offset relative to the GT state used to initialize ODE
    integration. The first supervised pose anchors the window and is excluded
    from the returned residuals because it cannot be changed by bias dynamics.
    """
    if R_ref.shape[0] < 2:
        raise ValueError("relative pose supervision needs at least two timestamps")

    R_ref_rel, p_ref_rel = _relative_pose(R_ref, p_ref)
    R_pred_rel, p_pred_rel = _relative_pose(R_pred, p_pred)

    if mode == "log":
        return se23_pose_residual(
            R_ref_rel[1:],
            p_ref_rel[1:],
            R_pred_rel[1:],
            p_pred_rel[1:],
        )
    if mode == "right":
        return pose_right_residual(
            R_ref_rel[1:],
            p_ref_rel[1:],
            R_pred_rel[1:],
            p_pred_rel[1:],
        )
    if mode == "global":
        return pose_global_residual(
            R_ref_rel[1:],
            p_ref_rel[1:],
            R_pred_rel[1:],
            p_pred_rel[1:],
        )
    if mode == "hybrid":
        log_residual = se23_pose_residual(
            R_ref_rel[1:],
            p_ref_rel[1:],
            R_pred_rel[1:],
            p_pred_rel[1:],
        )
        global_residual = pose_global_residual(
            R_ref_rel[1:],
            p_ref_rel[1:],
            R_pred_rel[1:],
            p_pred_rel[1:],
        )
        return torch.cat((log_residual[..., :3], global_residual[..., 3:]), dim=-1)
    raise ValueError("relative pose residual mode must be log, right, global, or hybrid")


def relative_se23_residual(
    X_ref: torch.Tensor,
    R_pred: torch.Tensor,
    solution: torch.Tensor,
    mode: str = "right",
) -> torch.Tensor:
    """Return SE_2(3) increment residuals relative to the first timestamp.

    The first supervised state anchors the local VO window and is excluded from
    the residual because bias dynamics cannot change the ODE initial state.
    """
    if X_ref.shape[0] < 2:
        raise ValueError("relative SE23 supervision needs at least two timestamps")

    R_ref_rel, v_ref_rel, p_ref_rel = _relative_se23_components(
        X_ref[..., :3, :3],
        X_ref[..., :3, 3],
        X_ref[..., :3, 4],
    )
    R_pred_rel, v_pred_rel, p_pred_rel = _relative_se23_components(
        R_pred,
        solution[..., 7:10],
        solution[..., 10:13],
    )

    if mode == "log":
        return _se23_components_log_residual(
            R_ref_rel[1:], v_ref_rel[1:], p_ref_rel[1:],
            R_pred_rel[1:], v_pred_rel[1:], p_pred_rel[1:],
        )
    if mode == "right":
        return _se23_components_right_residual(
            R_ref_rel[1:], v_ref_rel[1:], p_ref_rel[1:],
            R_pred_rel[1:], v_pred_rel[1:], p_pred_rel[1:],
        )
    if mode == "global":
        return _se23_components_global_residual(
            R_ref_rel[1:], v_ref_rel[1:], p_ref_rel[1:],
            R_pred_rel[1:], v_pred_rel[1:], p_pred_rel[1:],
        )
    if mode == "hybrid":
        log_residual = _se23_components_log_residual(
            R_ref_rel[1:], v_ref_rel[1:], p_ref_rel[1:],
            R_pred_rel[1:], v_pred_rel[1:], p_pred_rel[1:],
        )
        global_residual = _se23_components_global_residual(
            R_ref_rel[1:], v_ref_rel[1:], p_ref_rel[1:],
            R_pred_rel[1:], v_pred_rel[1:], p_pred_rel[1:],
        )
        return torch.cat((log_residual[..., :3], global_residual[..., 3:]), dim=-1)
    raise ValueError("relative SE23 residual mode must be log, right, global, or hybrid")


def _relative_se23_components(
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    R0 = R[0:1]
    v0 = v[0:1]
    p0 = p[0:1]
    R0T = R0.transpose(-1, -2)
    R_rel = torch.matmul(R0T, R)
    v_rel = torch.matmul(R0T, (v - v0).unsqueeze(-1)).squeeze(-1)
    p_rel = torch.matmul(R0T, (p - p0).unsqueeze(-1)).squeeze(-1)
    return R_rel, v_rel, p_rel


def _se23_components_log_residual(
    R_ref: torch.Tensor,
    v_ref: torch.Tensor,
    p_ref: torch.Tensor,
    R_pred: torch.Tensor,
    v_pred: torch.Tensor,
    p_pred: torch.Tensor,
) -> torch.Tensor:
    R_err = torch.matmul(R_ref.transpose(-1, -2), R_pred)
    v_err = torch.matmul(R_ref.transpose(-1, -2), (v_pred - v_ref).unsqueeze(-1)).squeeze(-1)
    p_err = torch.matmul(R_ref.transpose(-1, -2), (p_pred - p_ref).unsqueeze(-1)).squeeze(-1)
    X_err = Lie.SEn3fromSO3Rn(R_err, torch.cat((v_err, p_err), dim=-1))
    return Lie.SEn3log(X_err)


def _se23_components_right_residual(
    R_ref: torch.Tensor,
    v_ref: torch.Tensor,
    p_ref: torch.Tensor,
    R_pred: torch.Tensor,
    v_pred: torch.Tensor,
    p_pred: torch.Tensor,
) -> torch.Tensor:
    R_err = torch.matmul(R_ref, R_pred.transpose(-1, -2))
    v_err = v_ref - torch.matmul(R_err, v_pred.unsqueeze(-1)).squeeze(-1)
    p_err = p_ref - torch.matmul(R_err, p_pred.unsqueeze(-1)).squeeze(-1)
    X_err = Lie.SEn3fromSO3Rn(R_err, torch.cat((v_err, p_err), dim=-1))
    return Lie.SEn3log(X_err)


def _se23_components_global_residual(
    R_ref: torch.Tensor,
    v_ref: torch.Tensor,
    p_ref: torch.Tensor,
    R_pred: torch.Tensor,
    v_pred: torch.Tensor,
    p_pred: torch.Tensor,
) -> torch.Tensor:
    phi = Lie.SO3log(torch.matmul(R_pred.transpose(-1, -2), R_ref))
    dv = v_pred - v_ref
    dp = p_pred - p_ref
    return torch.cat((phi, dv, dp), dim=-1)


def se23_matrix_global_residual(X_ref: torch.Tensor, X_pred: torch.Tensor) -> torch.Tensor:
    R_ref = X_ref[..., :3, :3]
    R_pred = X_pred[..., :3, :3]
    phi = Lie.SO3log(torch.matmul(R_pred.transpose(-1, -2), R_ref))
    dv = X_pred[..., :3, 3] - X_ref[..., :3, 3]
    dp = X_pred[..., :3, 4] - X_ref[..., :3, 4]
    return torch.cat((phi, dv, dp), dim=-1)


def _relative_pose(R: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    R0 = R[0:1]
    p0 = p[0:1]
    R0T = R0.transpose(-1, -2)
    R_rel = torch.matmul(R0T, R)
    p_rel = torch.matmul(R0T, (p - p0).unsqueeze(-1)).squeeze(-1)
    return R_rel, p_rel


def se23_mse_loss(
    residual: torch.Tensor,
    rot_weight: float = 1.0,
    vel_weight: float = 1.0,
    pos_weight: float = 1.0,
    blocks: tuple[str, ...] = ("rot", "vel", "pos"),
) -> torch.Tensor:
    """Compute 0.5 * mean(||W r||^2) over selected SE_2(3) blocks."""
    weights = {
        "rot": rot_weight,
        "vel": vel_weight,
        "pos": pos_weight,
    }
    weighted = _select_and_weight_blocks(residual, SE23_BLOCKS, weights, blocks)
    return 0.5 * weighted.square().sum(dim=-1).mean()


def pose_mse_loss(
    residual: torch.Tensor,
    rot_weight: float = 1.0,
    pos_weight: float = 1.0,
    blocks: tuple[str, ...] = ("rot", "pos"),
) -> torch.Tensor:
    """Compute 0.5 * mean(||W r||^2) over selected pose blocks."""
    weights = {
        "rot": rot_weight,
        "pos": pos_weight,
    }
    weighted = _select_and_weight_blocks(residual, POSE_BLOCKS, weights, blocks)
    return 0.5 * weighted.square().sum(dim=-1).mean()


def se23_block_mse(residual: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-block mean squared residuals for logging."""
    return {
        name: residual[..., block].square().mean()
        for name, block in SE23_BLOCKS.items()
    }


def pose_block_mse(residual: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-block mean squared residuals for logging."""
    return {
        name: residual[..., block].square().mean()
        for name, block in POSE_BLOCKS.items()
    }


def _select_and_weight_blocks(
    residual: torch.Tensor,
    block_slices: dict[str, slice],
    weights: dict[str, float],
    blocks: tuple[str, ...],
) -> torch.Tensor:
    if residual.shape[-1] != max(block.stop for block in block_slices.values()):
        raise ValueError(
            f"Expected residual last dim {max(block.stop for block in block_slices.values())}, "
            f"got {residual.shape[-1]}"
        )
    if not blocks:
        raise ValueError("At least one residual block must be selected")

    pieces = []
    for name in blocks:
        if name not in block_slices:
            raise ValueError(f"Unknown residual block '{name}'")
        block = residual[..., block_slices[name]]
        weight = torch.as_tensor(weights[name], device=residual.device, dtype=residual.dtype)
        pieces.append(block * weight)
    return torch.cat(pieces, dim=-1)
