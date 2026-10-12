from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

BIASDY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BIASDY_DIR not in sys.path:
    sys.path.insert(0, BIASDY_DIR)

try:
    import lie_algebra as Lie
except ImportError:
    from BiasDy import lie_algebra as Lie

try:
    from plato.covariance import (
        build_se23_precision,
        contract_interval_covariance_state_forward_sensitivity_chunked,
        preintegrate_interval_covariance,
        preintegrate_interval_covariance_forward_sensitivity,
        preintegrate_interval_covariance_state_vjp,
        preintegrate_supervision_covariances,
        se23_Gd,
        se23_Gd_left_tangent_sensitivity,
        se23_phi,
        safe_cholesky,
    )
    from plato.nll import (
        PlatoNoiseParams,
        full_state_nees,
        full_state_nll,
        pose_nees,
        pose_nll,
        sparse_full_state_nees,
        sparse_full_state_nll,
        sparse_pose_nees,
        sparse_pose_nll,
        sparse_rotation_nees,
        sparse_rotation_nll,
        se23_right_jacobian_inverse_logabsdet,
        so3_right_jacobian_inverse_logabsdet,
    )
except ImportError:
    from .covariance import (
        build_se23_precision,
        contract_interval_covariance_state_forward_sensitivity_chunked,
        preintegrate_interval_covariance,
        preintegrate_interval_covariance_forward_sensitivity,
        preintegrate_interval_covariance_state_vjp,
        preintegrate_supervision_covariances,
        se23_Gd,
        se23_Gd_left_tangent_sensitivity,
        se23_phi,
        safe_cholesky,
    )
    from .nll import (
        PlatoNoiseParams,
        full_state_nees,
        full_state_nll,
        pose_nees,
        pose_nll,
        sparse_full_state_nees,
        sparse_full_state_nll,
        sparse_pose_nees,
        sparse_pose_nll,
        sparse_rotation_nees,
        sparse_rotation_nll,
        se23_right_jacobian_inverse_logabsdet,
        so3_right_jacobian_inverse_logabsdet,
    )


def _make_test_trajectory(T: int = 8, B: int = 2):
    torch.manual_seed(19)
    t = 0.005 * torch.arange(T, dtype=torch.float64)
    R = Lie.SO3exp(0.01 * torch.randn(T, B, 3, dtype=torch.float64))
    v = torch.randn(T, B, 3, dtype=torch.float64)
    p = torch.randn(T, B, 3, dtype=torch.float64)
    return R, v, p, t


def test_phi_and_gd_shapes():
    R, v, p, t = _make_test_trajectory()
    Phi = se23_phi(t[1] - t[0], device=R.device, dtype=R.dtype)
    Gd = se23_Gd(R[0], v[0], p[0], t[1] - t[0])
    assert Phi.shape == (9, 9)
    assert Gd.shape == (2, 9, 6)


def test_gd_left_tangent_sensitivity_matches_autograd():
    R, v, p, t = _make_test_trajectory(T=3, B=1)
    R0 = R[0, 0]
    v0 = v[0, 0]
    p0 = p[0, 0]
    dt = t[1] - t[0]
    X0 = Lie.SEn3fromSO3Rn(R0, torch.cat([v0, p0], dim=-1))

    def perturbed_Gd(delta):
        X = torch.matmul(Lie.SEn3exp(delta), X0)
        return se23_Gd(X[:3, :3], X[:3, 3], X[:3, 4], dt)

    autograd_jacobian = torch.autograd.functional.jacobian(
        perturbed_Gd,
        torch.zeros(9, dtype=R.dtype),
    ).movedim(-1, 0)
    analytic_jacobian = se23_Gd_left_tangent_sensitivity(R0, v0, p0, dt)
    assert torch.allclose(
        analytic_jacobian, autograd_jacobian, atol=1e-12, rtol=1e-10
    )


def test_covariance_forward_sensitivity_matches_autograd():
    R_batch, v_batch, p_batch, t = _make_test_trajectory(T=3, B=1)
    R = R_batch[:, 0]
    v = v_batch[:, 0]
    p = p_batch[:, 0]
    noise = PlatoNoiseParams(dtype=R.dtype, device=R.device)
    result = preintegrate_interval_covariance_forward_sensitivity(
        R,
        v,
        p,
        t,
        0,
        2,
        noise.Qn().detach(),
        Qn_sensitivity=noise.Qn_log_sigma_sensitivity().detach(),
        include_state_sensitivity=True,
    )

    def covariance_from_log_sigma(log_sigma):
        variance_a = log_sigma[0].exp().square().expand(3)
        variance_g = log_sigma[1].exp().square().expand(3)
        Qn = torch.diag(torch.cat([variance_a, variance_g], dim=0))
        return preintegrate_interval_covariance(R, v, p, t, 0, 2, Qn)[1]

    noise_jacobian = torch.autograd.functional.jacobian(
        covariance_from_log_sigma,
        noise.log_sigma_vector().detach().requires_grad_(True),
    ).movedim(-1, 0)
    assert result.noise is not None
    assert torch.allclose(result.noise, noise_jacobian, atol=1e-12, rtol=1e-9)

    X = Lie.SEn3fromSO3Rn(R[:2], torch.cat([v[:2], p[:2]], dim=-1))

    def covariance_from_state_delta(delta):
        X_perturbed = torch.matmul(Lie.SEn3exp(delta), X)
        R_perturbed = torch.cat([X_perturbed[..., :3, :3], R[2:]], dim=0)
        v_perturbed = torch.cat([X_perturbed[..., :3, 3], v[2:]], dim=0)
        p_perturbed = torch.cat([X_perturbed[..., :3, 4], p[2:]], dim=0)
        return preintegrate_interval_covariance(
            R_perturbed,
            v_perturbed,
            p_perturbed,
            t,
            0,
            2,
            noise.Qn().detach(),
        )[1]

    state_jacobian = torch.autograd.functional.jacobian(
        covariance_from_state_delta,
        torch.zeros(2, 9, dtype=R.dtype, requires_grad=True),
    ).permute(2, 3, 0, 1)
    assert result.state is not None
    assert torch.allclose(result.state, state_jacobian, atol=1e-12, rtol=1e-9)


def test_covariance_state_vjp_matches_forward_sensitivity_and_autograd():
    """Validate the analytic VJP against both independent reference paths."""
    torch.manual_seed(23)
    T, B = 6, 2
    R, v, p, t = _make_test_trajectory(T=T, B=B)
    start_idx, end_idx = 1, 5
    noise = PlatoNoiseParams(dtype=R.dtype, device=R.device)
    Qn = noise.Qn().detach()
    covariance_cotangent = torch.randn(B, 9, 9, dtype=R.dtype)

    forward = preintegrate_interval_covariance_forward_sensitivity(
        R,
        v,
        p,
        t,
        start_idx,
        end_idx,
        Qn,
        include_state_sensitivity=True,
    )
    assert forward.state is not None
    forward_vjp = torch.einsum(
        "sk...ij,...ij->s...k",
        forward.state,
        covariance_cotangent,
    )
    chunked_forward = {
        chunk_size: contract_interval_covariance_state_forward_sensitivity_chunked(
            R,
            v,
            p,
            t,
            start_idx,
            end_idx,
            Qn,
            covariance_cotangent,
            chunk_size=chunk_size,
        )
        for chunk_size in (1, 2, 3, 8)
    }

    analytic_vjp = preintegrate_interval_covariance_state_vjp(
        R,
        v,
        p,
        t,
        start_idx,
        end_idx,
        Qn,
        covariance_cotangent,
    )

    interval_steps = end_idx - start_idx
    X_interval = Lie.SEn3fromSO3Rn(
        R[start_idx:end_idx],
        torch.cat(
            [v[start_idx:end_idx], p[start_idx:end_idx]],
            dim=-1,
        ),
    )
    delta = torch.zeros(
        interval_steps,
        B,
        9,
        dtype=R.dtype,
        requires_grad=True,
    )
    X_perturbed = torch.matmul(Lie.SEn3exp(delta), X_interval)
    R_perturbed = R.clone()
    v_perturbed = v.clone()
    p_perturbed = p.clone()
    R_perturbed[start_idx:end_idx] = X_perturbed[..., :3, :3]
    v_perturbed[start_idx:end_idx] = X_perturbed[..., :3, 3]
    p_perturbed[start_idx:end_idx] = X_perturbed[..., :3, 4]
    covariance_autograd = preintegrate_interval_covariance(
        R_perturbed,
        v_perturbed,
        p_perturbed,
        t,
        start_idx,
        end_idx,
        Qn,
    )[1]
    scalar = (covariance_autograd * covariance_cotangent).sum()
    autograd_vjp = torch.autograd.grad(scalar, delta)[0]

    assert torch.allclose(
        analytic_vjp,
        forward_vjp,
        atol=2e-12,
        rtol=2e-10,
    )
    assert torch.allclose(
        analytic_vjp,
        autograd_vjp,
        atol=2e-12,
        rtol=2e-10,
    )
    for chunk_size, chunked_gradient in chunked_forward.items():
        assert torch.allclose(
            chunked_gradient,
            forward_vjp,
            atol=2e-12,
            rtol=2e-10,
        ), chunk_size
        assert torch.allclose(
            chunked_gradient,
            autograd_vjp,
            atol=2e-12,
            rtol=2e-10,
        ), chunk_size


def test_forward_sensitivity_nll_gradient_matches_reverse_ad():
    try:
        from plato.training import (
            _apply_se23_left_perturbation,
            _preintegrate_nll_covariances,
        )
    except ImportError:
        from .training import (
            _apply_se23_left_perturbation,
            _preintegrate_nll_covariances,
        )

    torch.manual_seed(29)
    T = 7
    B = 2
    dtype = torch.float64
    t = 0.005 * torch.arange(T, dtype=dtype)
    R = Lie.SO3exp(0.05 * torch.randn(T, B, 3, dtype=dtype))
    solution = torch.zeros(T, B, 16, dtype=dtype)
    solution[..., 7:13] = 0.1 * torch.randn(T, B, 6, dtype=dtype)
    supervised_indices = torch.tensor([0, 2, 4, 6])
    residual = 1e-3 * torch.randn(4, B, 9, dtype=dtype)

    def run(covariance_gradient_mode):
        noise = PlatoNoiseParams(dtype=dtype)
        delta = torch.zeros(T, B, 9, dtype=dtype, requires_grad=True)
        R_perturbed, solution_perturbed = _apply_se23_left_perturbation(
            R, solution, delta
        )
        Phi_list, Q_list = _preintegrate_nll_covariances(
            R_perturbed,
            solution_perturbed[..., 7:10],
            solution_perturbed[..., 10:13],
            t,
            supervised_indices,
            noise,
            SimpleNamespace(covariance_gradient_mode=covariance_gradient_mode),
            covariance_nominal=(solution, R),
            covariance_state_perturbation=delta,
        )
        loss = sparse_full_state_nll(
            residual,
            Phi_list,
            Q_list,
            include_prior=False,
            include_logdet=True,
            jitter=0.0,
            reduction="mean",
            time_first=True,
        )
        loss.backward()
        return (
            loss.detach(),
            delta.grad.detach(),
            noise.log_sigma_a.grad.detach(),
            noise.log_sigma_g.grad.detach(),
        )

    reverse = run("autograd")
    forward = run("forward_sensitivity")
    for reverse_value, forward_value in zip(reverse, forward):
        assert torch.allclose(
            forward_value, reverse_value, atol=1e-9, rtol=1e-8
        )


def test_se23_logdet_h_matches_matrix_value_and_gradient():
    residual = torch.tensor(
        [
            [0.07, -0.03, 0.02, 0.4, -0.2, 0.1, -0.3, 0.5, 0.8],
            [-0.4, 0.2, 0.1, -0.6, 0.3, 0.7, 0.2, -0.1, 0.9],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    closed_form = se23_right_jacobian_inverse_logabsdet(residual)
    H = -Lie.SEn3rightJaco_inv(residual)
    matrix_form = torch.linalg.slogdet(H).logabsdet
    assert torch.allclose(closed_form, matrix_form, atol=2e-14, rtol=2e-13)

    closed_grad = torch.autograd.grad(closed_form.sum(), residual, retain_graph=True)[0]
    matrix_grad = torch.autograd.grad(matrix_form.sum(), residual)[0]
    assert torch.allclose(closed_grad, matrix_grad, atol=2e-13, rtol=2e-12)
    assert torch.equal(closed_grad[..., 3:], torch.zeros_like(closed_grad[..., 3:]))

    zero = torch.zeros(1, 9, dtype=torch.float64, requires_grad=True)
    zero_value = se23_right_jacobian_inverse_logabsdet(zero)
    zero_grad = torch.autograd.grad(zero_value.sum(), zero)[0]
    assert torch.equal(zero_value, torch.zeros_like(zero_value))
    assert torch.equal(zero_grad, torch.zeros_like(zero_grad))


def test_sparse_full_state_logdet_h_toggle_and_covariance_gradient_invariance():
    torch.manual_seed(41)
    dtype = torch.float64
    Phi_list = torch.eye(9, dtype=dtype).expand(3, 2, 9, 9).clone()
    Q = (0.1 * torch.eye(9, dtype=dtype)).expand(3, 2, 9, 9).clone()
    Q.requires_grad_(True)
    residual = (0.1 * torch.randn(4, 2, 9, dtype=dtype)).requires_grad_(True)

    without_h = sparse_full_state_nll(
        residual,
        Phi_list,
        Q,
        include_logdet=True,
        include_logdet_h=False,
        reduction="none",
        time_first=True,
    )
    with_h = sparse_full_state_nll(
        residual,
        Phi_list,
        Q,
        include_logdet=True,
        include_logdet_h=True,
        reduction="none",
        time_first=True,
    )
    expected_h = se23_right_jacobian_inverse_logabsdet(residual).sum(dim=0)
    assert torch.allclose(with_h - without_h, expected_h, atol=1e-12, rtol=1e-12)

    grad_r_with, grad_q_with = torch.autograd.grad(
        with_h.sum(), (residual, Q), retain_graph=True
    )
    grad_r_without, grad_q_without = torch.autograd.grad(
        without_h.sum(), (residual, Q), retain_graph=True
    )
    grad_r_expected = torch.autograd.grad(expected_h.sum(), residual)[0]
    assert torch.allclose(
        grad_r_with - grad_r_without, grad_r_expected, atol=1e-10, rtol=1e-10
    )
    assert torch.allclose(grad_q_with, grad_q_without, atol=1e-12, rtol=1e-12)


def test_precision_spd_and_nll_finite():
    R, v, p, t = _make_test_trajectory()
    noise = PlatoNoiseParams(dtype=R.dtype, device=R.device)
    supervised_indices = torch.tensor([0, 2, 5, 7])
    Phi_list, Q_list = preintegrate_supervision_covariances(
        R, v, p, t, supervised_indices, noise.Qn()
    )
    Lambda = build_se23_precision(Phi_list, Q_list, prior_std=1e-3)
    safe_cholesky(Q_list)
    safe_cholesky(Lambda)

    residual_full = 1e-3 * torch.randn(4, 2, 9, dtype=R.dtype)
    residual_pose = residual_full[..., [0, 1, 2, 6, 7, 8]]
    loss_full = full_state_nll(residual_full, Lambda, reduction="mean")
    loss_full_batch_first = full_state_nll(
        residual_full.transpose(0, 1), Lambda, reduction="mean", time_first=False
    )
    loss_pose = pose_nll(residual_pose, Lambda, reduction="mean")
    assert torch.isfinite(loss_full)
    assert torch.isfinite(loss_full_batch_first)
    assert torch.isfinite(loss_pose)


def test_nees_matches_quadratic_nll_terms():
    R, v, p, t = _make_test_trajectory()
    noise = PlatoNoiseParams(dtype=R.dtype, device=R.device)
    supervised_indices = torch.tensor([0, 2, 5, 7])
    Phi_list, Q_list = preintegrate_supervision_covariances(
        R, v, p, t, supervised_indices, noise.Qn()
    )
    Lambda = build_se23_precision(Phi_list, Q_list, prior_std=1e-3)

    residual_full = 1e-3 * torch.randn(4, 2, 9, dtype=R.dtype)
    residual_pose = residual_full[..., [0, 1, 2, 6, 7, 8]]

    full_quad = full_state_nll(
        residual_full,
        Lambda,
        reduction="none",
        time_first=True,
        include_logdet=False,
    )
    full_metric = full_state_nees(
        residual_full,
        Lambda,
        reduction="none",
        time_first=True,
    )
    pose_quad = pose_nll(
        residual_pose,
        Lambda,
        reduction="none",
        time_first=True,
        include_logdet=False,
    )
    pose_metric = pose_nees(
        residual_pose,
        Lambda,
        reduction="none",
        time_first=True,
    )
    sparse_full_quad = sparse_full_state_nll(
        residual_full,
        Phi_list,
        Q_list,
        reduction="none",
        time_first=True,
        include_logdet=False,
        include_logdet_h=False,
    )
    sparse_full_metric = sparse_full_state_nees(
        residual_full,
        Phi_list,
        Q_list,
        reduction="none",
        time_first=True,
    )
    sparse_pose_quad = sparse_pose_nll(
        residual_pose,
        Phi_list,
        Q_list,
        reduction="none",
        time_first=True,
        include_logdet=False,
    )
    sparse_pose_metric = sparse_pose_nees(
        residual_pose,
        Phi_list,
        Q_list,
        reduction="none",
        time_first=True,
    )

    assert torch.allclose(full_metric, 2.0 * full_quad, atol=1e-8, rtol=1e-8)
    assert torch.allclose(pose_metric, 2.0 * pose_quad, atol=1e-8, rtol=1e-8)
    assert torch.allclose(sparse_full_metric, 2.0 * sparse_full_quad, atol=1e-8, rtol=1e-8)
    assert torch.allclose(sparse_pose_metric, 2.0 * sparse_pose_quad, atol=1e-8, rtol=1e-8)


def test_nll_gradients():
    R, v, p, t = _make_test_trajectory()
    noise = PlatoNoiseParams(dtype=R.dtype, device=R.device)
    residual = torch.nn.Parameter(1e-3 * torch.randn(4, 2, 9, dtype=R.dtype))
    supervised_indices = torch.tensor([0, 2, 5, 7])

    Phi_list, Q_list = preintegrate_supervision_covariances(
        R, v, p, t, supervised_indices, noise.Qn()
    )
    Lambda = build_se23_precision(Phi_list, Q_list, prior_std=1e-3)
    loss = full_state_nll(residual, Lambda, reduction="mean")
    loss.backward()

    assert residual.grad is not None
    assert noise.log_sigma_a.grad is not None
    assert noise.log_sigma_g.grad is not None


def test_sparse_full_state_nll_finite_and_gradients():
    R, v, p, t = _make_test_trajectory()
    noise = PlatoNoiseParams(dtype=R.dtype, device=R.device)
    residual = torch.nn.Parameter(1e-3 * torch.randn(4, 2, 9, dtype=R.dtype))
    supervised_indices = torch.tensor([0, 2, 5, 7])

    Phi_list, Q_list = preintegrate_supervision_covariances(
        R, v, p, t, supervised_indices, noise.Qn()
    )
    loss = sparse_full_state_nll(
        residual,
        Phi_list,
        Q_list,
        reduction="mean",
        time_first=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert residual.grad is not None
    assert noise.log_sigma_a.grad is not None
    assert noise.log_sigma_g.grad is not None


def test_sparse_rotation_marginal_is_rotation_only():
    R, v, p, t = _make_test_trajectory()
    noise = PlatoNoiseParams(dtype=R.dtype, device=R.device)
    residual = torch.nn.Parameter(1e-3 * torch.randn(4, 2, 9, dtype=R.dtype))
    supervised_indices = torch.tensor([0, 2, 5, 7])

    Phi_list, Q_list = preintegrate_supervision_covariances(
        R, v, p, t, supervised_indices, noise.Qn()
    )
    loss = sparse_rotation_nll(
        residual,
        Phi_list,
        Q_list,
        reduction="mean",
        time_first=True,
    )
    nees = sparse_rotation_nees(
        residual,
        Phi_list,
        Q_list,
        reduction="mean",
        time_first=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(nees)
    assert residual.grad is not None
    assert torch.count_nonzero(residual.grad[..., 3:]) == 0
    assert noise.log_sigma_g.grad is not None
    assert noise.log_sigma_g.grad.abs().max() > 0
    assert noise.log_sigma_a.grad is not None
    assert noise.log_sigma_a.grad.abs().max() < 1e-12

    phi = residual.detach()[..., :3]
    padded = torch.zeros_like(residual.detach())
    padded[..., :3] = phi
    assert torch.allclose(
        se23_right_jacobian_inverse_logabsdet(padded),
        3.0 * so3_right_jacobian_inverse_logabsdet(phi),
        rtol=1e-13,
        atol=1e-13,
    )


def test_sparse_pose_nll_matches_dense_woodbury():
    R, v, p, t = _make_test_trajectory()
    noise = PlatoNoiseParams(dtype=R.dtype, device=R.device)
    residual = torch.nn.Parameter(1e-3 * torch.randn(4, 2, 6, dtype=R.dtype))
    supervised_indices = torch.tensor([0, 2, 5, 7])

    Phi_list, Q_list = preintegrate_supervision_covariances(
        R, v, p, t, supervised_indices, noise.Qn()
    )
    Lambda = build_se23_precision(Phi_list, Q_list, prior_std=1e-3)
    dense = pose_nll(
        residual,
        Lambda,
        meas_sigma_rot=1e-2,
        meas_sigma_pos=5e-2,
        reduction="mean",
        time_first=True,
    )
    sparse = sparse_pose_nll(
        residual,
        Phi_list,
        Q_list,
        meas_sigma_rot=1e-2,
        meas_sigma_pos=5e-2,
        prior_std=1e-3,
        reduction="mean",
        time_first=True,
    )
    sparse.backward()

    assert torch.isfinite(sparse)
    assert torch.allclose(sparse, dense, atol=1e-4, rtol=1e-4)
    assert residual.grad is not None
    assert noise.log_sigma_a.grad is not None
    assert noise.log_sigma_g.grad is not None


def test_zero_jitter_rejects_one_step_singular_covariance():
    R, v, p, t = _make_test_trajectory()
    noise = PlatoNoiseParams(dtype=R.dtype, device=R.device)
    _, Q_list = preintegrate_supervision_covariances(
        R,
        v,
        p,
        t,
        torch.tensor([0, 1, 3]),
        noise.Qn(),
    )
    try:
        safe_cholesky(Q_list)
    except RuntimeError as error:
        assert "fixed jitter disabled" in str(error)
    else:
        raise AssertionError("One-step raw covariance unexpectedly passed Cholesky")


def test_training_gt_nll_is_two_step_float64_with_float32_gradients():
    try:
        from plato.training import _nll_loss_for_batch
    except ImportError:
        from .training import _nll_loss_for_batch

    T = 8
    B = 2
    t = 0.005 * torch.arange(T, dtype=torch.float32)
    R_sol = torch.eye(3, dtype=torch.float32).repeat(T, B, 1, 1)
    solution = torch.zeros(T, B, 16, dtype=torch.float32, requires_grad=True)
    X_ref = Lie.SEn3fromSO3Rn(
        R_sol,
        torch.zeros(T, B, 6, dtype=torch.float32),
    )
    batch = SimpleNamespace(
        supervised_indices=torch.arange(T, dtype=torch.long),
        t_odeint=t,
        t_state=t,
        X_ref=X_ref,
    )
    config = SimpleNamespace(
        supervision="gt_se23",
        nll_gt_sup_stride=2,
        cov_jitter=0.0,
        nll_gt_mode="sparse",
        nll_gt_residual_mode="right",
        nll_gt_include_logdet_h=True,
        prior_std=1e-4,
        meas_sigma_rot_gt=1e-3,
        meas_sigma_vel_gt=1e-2,
        meas_sigma_pos_gt=1e-2,
        nll_mse_anchor_weight=0.0,
    )
    noise = PlatoNoiseParams(dtype=torch.float32)

    sparse_loss, sparse_metrics = _nll_loss_for_batch(
        batch, solution, R_sol, config, noise
    )
    assert sparse_loss.dtype == torch.float64
    assert sparse_metrics["nll"].dtype == torch.float64
    assert sparse_metrics["logdet_h"].dtype == torch.float64
    assert sparse_metrics["nees_dof"].item() == 27.0  # 3 two-step intervals * 9.
    assert torch.isfinite(sparse_loss)

    config.nll_gt_mode = "woodbury"
    woodbury_loss, woodbury_metrics = _nll_loss_for_batch(
        batch, solution, R_sol, config, noise
    )
    assert woodbury_loss.dtype == torch.float64
    assert woodbury_metrics["nll"].dtype == torch.float64
    assert woodbury_metrics["nees_dof"].item() == 36.0  # 4 supervised states * 9.
    assert torch.isfinite(woodbury_loss)
    assert torch.equal(batch.supervised_indices, torch.arange(T, dtype=torch.long))

    (sparse_loss + woodbury_loss).backward()
    assert solution.grad is not None
    assert solution.grad.dtype == torch.float32
    assert noise.log_sigma_a.grad is not None
    assert noise.log_sigma_a.grad.dtype == torch.float32
    assert noise.log_sigma_g.grad is not None
    assert noise.log_sigma_g.grad.dtype == torch.float32


def test_streamed_interval_state_sensitivity_matches_materialized_path():
    try:
        from plato.training import (
            _apply_se23_left_perturbation,
            _nll_loss_for_batch,
            _sparse_gt_nll_streaming_covariance_double_adjoint_backward,
        )
    except ImportError:
        from .training import (
            _apply_se23_left_perturbation,
            _nll_loss_for_batch,
            _sparse_gt_nll_streaming_covariance_double_adjoint_backward,
        )

    T, B = 8, 2
    R, v, p, t = _make_test_trajectory(T=T, B=B)
    solution = torch.zeros(T, B, 16, dtype=torch.float64)
    solution[..., 3] = t.unsqueeze(-1)
    solution[..., 7:10] = v
    solution[..., 10:13] = p
    solution[..., 4:7] = 0.01 * torch.randn(T, B, 3, dtype=torch.float64)
    solution[..., 13:16] = 0.01 * torch.randn(T, B, 3, dtype=torch.float64)
    X_ref = Lie.SEn3fromSO3Rn(
        Lie.SO3exp(0.002 * torch.randn(T, B, 3, dtype=torch.float64)) @ R,
        torch.cat((v + 0.01, p - 0.02), dim=-1),
    )
    batch = SimpleNamespace(
        supervised_indices=torch.arange(T, dtype=torch.long),
        t_odeint=t,
        t_state=t,
        X_ref=X_ref,
    )
    config = SimpleNamespace(
        supervision="gt_se23",
        nll_gt_sup_stride=2,
        cov_jitter=0.0,
        nll_gt_mode="sparse",
        nll_gt_residual_mode="right",
        nll_gt_include_logdet_h=True,
        covariance_gradient_mode="forward_sensitivity",
        prior_std=1e-4,
        nll_mse_anchor_weight=0.0,
    )

    delta_materialized = torch.zeros(T, B, 9, dtype=torch.float64, requires_grad=True)
    R_materialized, solution_materialized = _apply_se23_left_perturbation(
        R, solution, delta_materialized
    )
    noise_materialized = PlatoNoiseParams(
        init_sigma_a=0.1,
        init_sigma_g=0.1,
        dtype=torch.float64,
    )
    loss_materialized, metrics_materialized = _nll_loss_for_batch(
        batch,
        solution_materialized,
        R_materialized,
        config,
        noise_materialized,
        covariance_nominal=(solution, R),
        covariance_state_perturbation=delta_materialized,
    )
    loss_materialized.backward()

    delta_streamed = torch.zeros(T, B, 9, dtype=torch.float64, requires_grad=True)
    R_streamed, solution_streamed = _apply_se23_left_perturbation(
        R, solution, delta_streamed
    )
    noise_streamed = PlatoNoiseParams(
        init_sigma_a=0.1,
        init_sigma_g=0.1,
        dtype=torch.float64,
    )
    loss_streamed, metrics_streamed, state_gradient_streamed = (
        _sparse_gt_nll_streaming_covariance_double_adjoint_backward(
            batch,
            solution_streamed,
            R_streamed,
            solution,
            R,
            delta_streamed,
            config,
            noise_streamed,
        )
    )

    assert torch.allclose(loss_streamed, loss_materialized, rtol=1e-13, atol=1e-13)
    state_difference = state_gradient_streamed - delta_materialized.grad
    assert state_difference.norm() / delta_materialized.grad.norm() < 2e-12, (
        float(state_difference.norm()),
        float(state_difference.abs().max()),
        float(delta_materialized.grad.norm()),
    )
    assert torch.allclose(
        noise_streamed.log_sigma_a.grad,
        noise_materialized.log_sigma_a.grad,
        rtol=2e-12,
        atol=2e-12,
    )
    assert torch.allclose(
        noise_streamed.log_sigma_g.grad,
        noise_materialized.log_sigma_g.grad,
        rtol=2e-12,
        atol=2e-12,
    )
    assert torch.allclose(
        metrics_streamed["nees"], metrics_materialized["nees"], rtol=1e-13, atol=1e-13
    )

    # Repeat the materialized-vs-streamed equivalence check for the autonomous
    # rotation marginal used by staged BG-only NLL training.
    config.nll_gt_blocks = "rot"
    delta_rot_materialized = torch.zeros(
        T, B, 9, dtype=torch.float64, requires_grad=True
    )
    R_rot_materialized, solution_rot_materialized = _apply_se23_left_perturbation(
        R, solution, delta_rot_materialized
    )
    noise_rot_materialized = PlatoNoiseParams(
        init_sigma_a=0.1,
        init_sigma_g=0.1,
        dtype=torch.float64,
    )
    loss_rot_materialized, metrics_rot_materialized = _nll_loss_for_batch(
        batch,
        solution_rot_materialized,
        R_rot_materialized,
        config,
        noise_rot_materialized,
        covariance_nominal=(solution, R),
        covariance_state_perturbation=delta_rot_materialized,
    )
    loss_rot_materialized.backward()

    delta_rot_streamed = torch.zeros(
        T, B, 9, dtype=torch.float64, requires_grad=True
    )
    R_rot_streamed, solution_rot_streamed = _apply_se23_left_perturbation(
        R, solution, delta_rot_streamed
    )
    noise_rot_streamed = PlatoNoiseParams(
        init_sigma_a=0.1,
        init_sigma_g=0.1,
        dtype=torch.float64,
    )
    loss_rot_streamed, metrics_rot_streamed, state_gradient_rot_streamed = (
        _sparse_gt_nll_streaming_covariance_double_adjoint_backward(
            batch,
            solution_rot_streamed,
            R_rot_streamed,
            solution,
            R,
            delta_rot_streamed,
            config,
            noise_rot_streamed,
        )
    )
    assert torch.allclose(
        loss_rot_streamed, loss_rot_materialized, rtol=1e-13, atol=1e-13
    )
    rot_state_difference = (
        state_gradient_rot_streamed - delta_rot_materialized.grad
    )
    assert rot_state_difference.norm() / delta_rot_materialized.grad.norm() < 2e-12, (
        float(rot_state_difference.norm()),
        float(rot_state_difference.abs().max()),
        float(delta_rot_materialized.grad.norm()),
    )
    assert torch.allclose(
        noise_rot_streamed.log_sigma_a.grad,
        noise_rot_materialized.log_sigma_a.grad,
        rtol=2e-12,
        atol=2e-12,
    )
    assert torch.allclose(
        noise_rot_streamed.log_sigma_g.grad,
        noise_rot_materialized.log_sigma_g.grad,
        rtol=2e-12,
        atol=2e-12,
    )
    assert torch.allclose(
        metrics_rot_streamed["nees"],
        metrics_rot_materialized["nees"],
        rtol=1e-13,
        atol=1e-13,
    )
    config.nll_gt_blocks = "full"

    # The actual alternating-NLL bias block freezes sigma_a/sigma_g. Verify
    # that skipping their unused sensitivities leaves the complete bias-state
    # gradient unchanged and produces no covariance-parameter gradients.
    delta_bias_only = torch.zeros(T, B, 9, dtype=torch.float64, requires_grad=True)
    R_bias_only, solution_bias_only = _apply_se23_left_perturbation(
        R, solution, delta_bias_only
    )
    noise_bias_only = PlatoNoiseParams(
        init_sigma_a=0.1,
        init_sigma_g=0.1,
        dtype=torch.float64,
    )
    for parameter in noise_bias_only.parameters():
        parameter.requires_grad_(False)
    loss_bias_only, metrics_bias_only, state_gradient_bias_only = (
        _sparse_gt_nll_streaming_covariance_double_adjoint_backward(
            batch,
            solution_bias_only,
            R_bias_only,
            solution,
            R,
            delta_bias_only,
            config,
            noise_bias_only,
        )
    )
    assert torch.allclose(loss_bias_only, loss_materialized, rtol=1e-13, atol=1e-13)
    bias_only_difference = state_gradient_bias_only - delta_materialized.grad
    assert bias_only_difference.norm() / delta_materialized.grad.norm() < 2e-12, (
        float(bias_only_difference.norm()),
        float(bias_only_difference.abs().max()),
        float(delta_materialized.grad.norm()),
    )
    assert noise_bias_only.log_sigma_a.grad is None
    assert noise_bias_only.log_sigma_g.grad is None
    assert torch.allclose(
        metrics_bias_only["nees"],
        metrics_materialized["nees"],
        rtol=1e-13,
        atol=1e-13,
    )


def main():
    test_phi_and_gd_shapes()
    test_gd_left_tangent_sensitivity_matches_autograd()
    test_covariance_forward_sensitivity_matches_autograd()
    test_covariance_state_vjp_matches_forward_sensitivity_and_autograd()
    test_forward_sensitivity_nll_gradient_matches_reverse_ad()
    test_se23_logdet_h_matches_matrix_value_and_gradient()
    test_sparse_full_state_logdet_h_toggle_and_covariance_gradient_invariance()
    test_precision_spd_and_nll_finite()
    test_nees_matches_quadratic_nll_terms()
    test_nll_gradients()
    test_sparse_full_state_nll_finite_and_gradients()
    test_sparse_rotation_marginal_is_rotation_only()
    test_sparse_pose_nll_matches_dense_woodbury()
    test_zero_jitter_rejects_one_step_singular_covariance()
    test_training_gt_nll_is_two_step_float64_with_float32_gradients()
    test_streamed_interval_state_sensitivity_matches_materialized_path()
    print("PLATO NLL smoke tests passed")


if __name__ == "__main__":
    main()
