from __future__ import annotations

import os
import sys

import torch

BIASDY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BIASDY_DIR not in sys.path:
    sys.path.insert(0, BIASDY_DIR)

try:
    import lie_algebra as Lie
    from network_for_BiadDy import ba_func_net, bw_func_net
except ImportError:  # Support `python -m BiasDy.plato.loss_smoke_test`.
    from BiasDy import lie_algebra as Lie
    from BiasDy.network_for_BiadDy import ba_func_net, bw_func_net

try:
    from plato.analytic_gradient import (
        adjoint_bias_gradient,
        invariant_transition_matrices,
        weighted_se23_mse_state_gradient,
    )
    from plato.losses import (
        pose_mse_loss,
        relative_pose_residual,
        se23_gt_right_residual,
        se23_log_residual,
        se23_mse_loss,
        se23_pose_residual,
    )
    from plato.so3_analytic_gradient import (
        so3_gyro_bias_gradient,
        so3_invariant_residual,
        weighted_so3_mse_state_gradient,
    )
except ImportError:
    from .analytic_gradient import (
        adjoint_bias_gradient,
        invariant_transition_matrices,
        weighted_se23_mse_state_gradient,
    )
    from .losses import (
        pose_mse_loss,
        relative_pose_residual,
        se23_gt_right_residual,
        se23_log_residual,
        se23_mse_loss,
        se23_pose_residual,
    )
    from .so3_analytic_gradient import (
        so3_gyro_bias_gradient,
        so3_invariant_residual,
        weighted_so3_mse_state_gradient,
    )


def _make_solution_from_x(v: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    solution = torch.zeros(*v.shape[:-1], 16, device=v.device, dtype=v.dtype)
    solution[..., 7:10] = v
    solution[..., 10:13] = p
    return solution


def _network_signature(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(param.shape) for name, param in module.named_parameters()}


def test_se23_residual_zero():
    torch.manual_seed(7)
    shape = (4, 3)
    phi = 0.05 * torch.randn(*shape, 3)
    R = Lie.SO3exp(phi)
    v = torch.randn(*shape, 3)
    p = torch.randn(*shape, 3)
    X_ref = Lie.SEn3fromSO3Rn(R, torch.cat([v, p], dim=-1))
    solution = _make_solution_from_x(v, p)

    residual = se23_log_residual(X_ref, R, solution)
    assert residual.abs().max() < 1e-5


def test_se23_residual_small_perturbation():
    torch.manual_seed(11)
    shape = (2, 3)
    phi = 0.05 * torch.randn(*shape, 3)
    R_ref = Lie.SO3exp(phi)
    v_ref = torch.randn(*shape, 3)
    p_ref = torch.randn(*shape, 3)
    X_ref = Lie.SEn3fromSO3Rn(R_ref, torch.cat([v_ref, p_ref], dim=-1))

    delta = 1e-3 * torch.randn(*shape, 9)
    X_pred = torch.matmul(X_ref, Lie.SEn3exp(delta))
    R_pred = X_pred[..., :3, :3]
    v_pred = X_pred[..., :3, 3]
    p_pred = X_pred[..., :3, 4]
    solution = _make_solution_from_x(v_pred, p_pred)

    residual = se23_log_residual(X_ref, R_pred, solution)
    assert (residual - delta).abs().max() < 2e-5


def test_se23_gt_right_residual_matches_group_error():
    torch.manual_seed(19)
    shape = (2, 3)
    R_ref = Lie.SO3exp(0.1 * torch.randn(*shape, 3))
    v_ref = torch.randn(*shape, 3)
    p_ref = torch.randn(*shape, 3)
    X_ref = Lie.SEn3fromSO3Rn(R_ref, torch.cat([v_ref, p_ref], dim=-1))

    delta = 1e-3 * torch.randn(*shape, 9)
    X_pred = torch.matmul(Lie.SEn3inverse(Lie.SEn3exp(delta)), X_ref)
    solution = _make_solution_from_x(X_pred[..., :3, 3], X_pred[..., :3, 4])

    residual = se23_gt_right_residual(X_ref, X_pred[..., :3, :3], solution)
    direct = Lie.SEn3log(torch.matmul(X_ref, Lie.SEn3inverse(X_pred)))
    assert (residual - direct).abs().max() < 2e-5
    assert (residual - delta).abs().max() < 2e-5


def test_pose_residual_zero():
    torch.manual_seed(13)
    shape = (5, 2)
    R = Lie.SO3exp(0.05 * torch.randn(*shape, 3))
    p = torch.randn(*shape, 3)

    residual = se23_pose_residual(R, p, R, p)
    assert residual.abs().max() < 1e-5


def test_relative_pose_residual_zero_and_shape():
    torch.manual_seed(17)
    shape = (6, 3)
    R0 = Lie.SO3exp(0.05 * torch.randn(*shape, 3))
    p0 = torch.randn(*shape, 3)
    R_offset = Lie.SO3exp(torch.tensor([0.2, -0.1, 0.05]))
    p_offset = torch.tensor([1.0, -0.4, 0.3])
    R_shifted = torch.matmul(R_offset.unsqueeze(0).unsqueeze(0), R0)
    p_shifted = torch.matmul(
        R_offset,
        p0.unsqueeze(-1),
    ).squeeze(-1) + p_offset

    residual = relative_pose_residual(R_shifted, p_shifted, R0, p0, mode="hybrid")
    assert residual.shape == (shape[0] - 1, shape[1], 6)
    assert residual.abs().max() < 1e-5


def test_mse_block_selection():
    residual = torch.ones(2, 3, 9)
    rot_loss = se23_mse_loss(residual, blocks=("rot",))
    vp_loss = se23_mse_loss(residual, blocks=("vel", "pos"))
    pose_loss = pose_mse_loss(torch.ones(2, 3, 6), blocks=("pos",))

    assert torch.allclose(rot_loss, torch.tensor(1.5))
    assert torch.allclose(vp_loss, torch.tensor(3.0))
    assert torch.allclose(pose_loss, torch.tensor(1.5))


def test_weighted_se23_mse_state_gradient_matches_autograd():
    torch.manual_seed(23)
    residual = (0.01 * torch.randn(4, 2, 9)).requires_grad_()
    loss = se23_mse_loss(
        residual,
        rot_weight=0.0,
        vel_weight=1.0,
        pos_weight=10.0,
        blocks=("vel", "pos"),
    )
    loss.backward()
    analytic = weighted_se23_mse_state_gradient(
        residual.detach(),
        rot_weight=0.0,
        vel_weight=1.0,
        pos_weight=10.0,
        blocks=("vel", "pos"),
        residual_jacobian="identity",
    )
    assert torch.allclose(residual.grad, analytic, atol=1e-7, rtol=1e-5)


def test_weighted_se23_bch_state_gradient_matches_autograd():
    torch.manual_seed(41)
    residual0 = 0.02 * torch.randn(4, 2, 9)
    residual0[..., 0:3] += torch.tensor([0.03, -0.02, 0.01])
    delta = torch.zeros_like(residual0, requires_grad=True)

    residual = Lie.SEn3log(torch.matmul(Lie.SEn3exp(residual0), Lie.SEn3exp(-delta)))
    loss = se23_mse_loss(
        residual,
        rot_weight=0.5,
        vel_weight=1.0,
        pos_weight=10.0,
        blocks=("rot", "vel", "pos"),
    )
    loss.backward()
    analytic = weighted_se23_mse_state_gradient(
        residual0,
        rot_weight=0.5,
        vel_weight=1.0,
        pos_weight=10.0,
        blocks=("rot", "vel", "pos"),
        residual_jacobian="bch",
    )
    assert torch.allclose(delta.grad, analytic, atol=5e-5, rtol=5e-3)


def test_weighted_so3_mse_state_gradient_matches_autograd():
    torch.manual_seed(31)
    residual = (0.01 * torch.randn(5, 2, 3)).requires_grad_()
    loss = 0.5 * (2.0 * residual).square().sum(dim=-1).mean()
    loss.backward()
    analytic = weighted_so3_mse_state_gradient(
        residual.detach(),
        rot_weight=2.0,
        residual_jacobian="identity",
    )
    assert torch.allclose(residual.grad, analytic, atol=1e-7, rtol=1e-5)


def test_weighted_so3_bch_state_gradient_matches_autograd():
    torch.manual_seed(43)
    residual0 = 0.04 * torch.randn(5, 2, 3)
    residual0 += torch.tensor([0.03, -0.02, 0.01])
    delta = torch.zeros_like(residual0, requires_grad=True)

    residual = Lie.SO3log(torch.matmul(Lie.SO3exp(residual0), Lie.SO3exp(-delta)))
    loss = 0.5 * (1.7 * residual).square().sum(dim=-1).mean()
    loss.backward()
    analytic = weighted_so3_mse_state_gradient(
        residual0,
        rot_weight=1.7,
        residual_jacobian="bch",
    )
    assert torch.allclose(delta.grad, analytic, atol=5e-5, rtol=5e-3)


def test_se23_outer_adjoint_shapes():
    torch.manual_seed(29)
    T, B = 5, 2
    R = Lie.SO3exp(0.01 * torch.randn(T, B, 3))
    v = torch.randn(T, B, 3)
    p = torch.randn(T, B, 3)
    t = torch.arange(T, dtype=torch.float32) * 0.005
    F, G = invariant_transition_matrices(R, v, p, t=t)
    state_grad = torch.randn(T, B, 9)
    analytic = adjoint_bias_gradient(state_grad, F, G)
    assert analytic.lambda_state.shape == (T, B, 9)
    assert analytic.bias_gradient.shape == (T - 1, B, 6)
    assert analytic.F.shape[-2:] == (9, 9)
    assert analytic.G.shape == (T - 1, B, 9, 6)


def test_se23_bias_gradient_order_is_ba_then_bg():
    T, B = 2, 1
    dt = 0.005
    R = torch.eye(3).reshape(1, 1, 3, 3).repeat(T, B, 1, 1)
    v = torch.zeros(T, B, 3)
    p = torch.zeros(T, B, 3)
    t = torch.tensor([0.0, dt], dtype=torch.float32)
    F, G = invariant_transition_matrices(R, v, p, t=t)

    vel_grad = torch.tensor([0.2, -0.3, 0.5]).reshape(1, 3)
    state_grad = torch.zeros(T, B, 9)
    state_grad[-1, :, 3:6] = vel_grad
    analytic = adjoint_bias_gradient(state_grad, F, G)
    assert torch.allclose(analytic.bias_gradient[0, :, 0:3], -dt * vel_grad)
    assert torch.allclose(analytic.bias_gradient[0, :, 3:6], torch.zeros_like(vel_grad))

    rot_grad = torch.tensor([-0.4, 0.1, 0.3]).reshape(1, 3)
    state_grad.zero_()
    state_grad[-1, :, 0:3] = rot_grad
    analytic = adjoint_bias_gradient(state_grad, F, G)
    assert torch.allclose(analytic.bias_gradient[0, :, 0:3], torch.zeros_like(rot_grad))
    assert torch.allclose(analytic.bias_gradient[0, :, 3:6], -dt * rot_grad)


def test_so3_outer_adjoint_shapes():
    torch.manual_seed(37)
    T, B = 6, 3
    R_ref = Lie.SO3exp(0.01 * torch.randn(T, B, 3))
    R_pred = torch.matmul(Lie.SO3exp(0.001 * torch.randn(T, B, 3)), R_ref)
    t = torch.arange(T, dtype=torch.float32) * 0.005
    residual = so3_invariant_residual(R_pred, R_pred, invariant_error="right")
    assert residual.abs().max() < 1e-5
    analytic = so3_gyro_bias_gradient(
        R_pred,
        R_ref,
        t=t,
        rot_weight=1.5,
        residual_jacobian="left",
        invariant_error="right",
    )
    assert analytic.lambda_state.shape == (T, B, 3)
    assert analytic.bias_gradient.shape == (T - 1, B, 3)
    assert analytic.state_gradient.shape == (T, B, 3)


def test_so3_sparse_outer_adjoint_scatter_and_dense_equivalence():
    torch.manual_seed(53)
    T, B = 7, 2
    R_pred = Lie.SO3exp(0.03 * torch.randn(T, B, 3))
    t = torch.arange(T, dtype=torch.float32) * 0.005
    supervised_indices = torch.tensor([0, 2, 6])
    R_ref_sparse = Lie.SO3exp(0.03 * torch.randn(3, B, 3))

    sparse = so3_gyro_bias_gradient(
        R_pred,
        R_ref_sparse,
        t=t,
        rot_weight=1.3,
        residual_jacobian="bch",
        invariant_error="right",
        supervised_indices=supervised_indices,
    )
    supervised_residual = so3_invariant_residual(
        R_pred[supervised_indices],
        R_ref_sparse,
        invariant_error="right",
    )
    supervised_gradient = weighted_so3_mse_state_gradient(
        supervised_residual,
        rot_weight=1.3,
        residual_jacobian="bch",
    )
    expected_state_gradient = torch.zeros(T, B, 3)
    expected_state_gradient.index_add_(
        0, supervised_indices, supervised_gradient
    )

    assert sparse.residual.shape == (3, B, 3)
    assert sparse.lambda_state.shape == (T, B, 3)
    assert sparse.bias_gradient.shape == (T - 1, B, 3)
    assert torch.allclose(
        sparse.state_gradient,
        expected_state_gradient,
        atol=1e-7,
        rtol=1e-6,
    )

    R_ref_dense = Lie.SO3exp(0.03 * torch.randn(T, B, 3))
    dense = so3_gyro_bias_gradient(
        R_pred,
        R_ref_dense,
        t=t,
        rot_weight=1.3,
        residual_jacobian="bch",
        invariant_error="right",
    )
    indexed_dense = so3_gyro_bias_gradient(
        R_pred,
        R_ref_dense,
        t=t,
        rot_weight=1.3,
        residual_jacobian="bch",
        invariant_error="right",
        supervised_indices=torch.arange(T),
    )
    assert torch.allclose(dense.residual, indexed_dense.residual)
    assert torch.allclose(dense.state_gradient, indexed_dense.state_gradient)
    assert torch.allclose(dense.lambda_state, indexed_dense.lambda_state)
    assert torch.allclose(dense.bias_gradient, indexed_dense.bias_gradient)


def test_network_shapes_unchanged():
    bw_expected = {
        "net.0.weight": (128, 9),
        "net.0.bias": (128,),
        "net.2.weight": (128, 128),
        "net.2.bias": (128,),
        "net.4.weight": (9, 128),
        "net.4.bias": (9,),
        "net2.0.weight": (512, 9),
        "net2.0.bias": (512,),
        "net2.2.weight": (9, 512),
        "net2.2.bias": (9,),
        "linear2.weight": (3, 9),
        "linear2.bias": (3,),
    }
    ba_expected = {
        "net.0.weight": (256, 9),
        "net.0.bias": (256,),
        "net.2.weight": (512, 256),
        "net.2.bias": (512,),
        "net.4.weight": (9, 512),
        "net.4.bias": (9,),
        "net2.0.weight": (256, 9),
        "net2.0.bias": (256,),
        "net2.2.weight": (256, 256),
        "net2.2.bias": (256,),
        "net2.4.weight": (9, 256),
        "net2.4.bias": (9,),
        "linear2.weight": (3, 9),
        "linear2.bias": (3,),
    }

    assert _network_signature(bw_func_net()) == bw_expected
    assert _network_signature(ba_func_net()) == ba_expected


def main():
    test_se23_residual_zero()
    test_se23_residual_small_perturbation()
    test_se23_gt_right_residual_matches_group_error()
    test_pose_residual_zero()
    test_relative_pose_residual_zero_and_shape()
    test_mse_block_selection()
    test_weighted_se23_mse_state_gradient_matches_autograd()
    test_weighted_se23_bch_state_gradient_matches_autograd()
    test_weighted_so3_mse_state_gradient_matches_autograd()
    test_weighted_so3_bch_state_gradient_matches_autograd()
    test_se23_outer_adjoint_shapes()
    test_se23_bias_gradient_order_is_ba_then_bg()
    test_so3_outer_adjoint_shapes()
    test_so3_sparse_outer_adjoint_scatter_and_dense_equivalence()
    test_network_shapes_unchanged()
    print("PLATO loss smoke tests passed")


if __name__ == "__main__":
    main()
