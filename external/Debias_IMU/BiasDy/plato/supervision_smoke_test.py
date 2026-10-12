from __future__ import annotations

import os
import sys

import torch

BIASDY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BIASDY_DIR not in sys.path:
    sys.path.insert(0, BIASDY_DIR)

try:
    import Interpolation
    import lie_algebra as Lie
except ImportError:
    from BiasDy import Interpolation
    from BiasDy import lie_algebra as Lie

try:
    from plato.supervision import (
        PoseTrajectory,
        align_vo_first_gt,
        build_gt_nll_supervised_indices,
        build_supervised_indices,
        get_gt_se23_supervision_batch,
        get_vo_pose_supervision_batch,
        get_vo_rate_gt_se23_supervision_batch,
        interpolate_pose,
        resolve_supervision_stride,
    )
except ImportError:
    from .supervision import (
        PoseTrajectory,
        align_vo_first_gt,
        build_gt_nll_supervised_indices,
        build_supervised_indices,
        get_gt_se23_supervision_batch,
        get_vo_pose_supervision_batch,
        get_vo_rate_gt_se23_supervision_batch,
        interpolate_pose,
        resolve_supervision_stride,
    )


class FakeEurocDataset:
    def __init__(self, mode: str = "train", batch_size: int = 5):
        self.dataset_name = "EUROC"
        self._mode = mode
        self._batch_size = batch_size
        self._loss_window = 16
        self._percent_for_val = 0.2
        self.dt = 0.005
        self.sequences = ["MH_fake"]
        T = 160
        t = self.dt * torch.arange(T, dtype=torch.float32)
        phi = torch.stack([0.01 * t, -0.02 * t, 0.005 * t], dim=-1)
        R = Lie.SO3exp(phi)
        v = torch.stack([0.1 + 0.01 * t, -0.2 + 0.02 * t, 0.05 * torch.ones_like(t)], dim=-1)
        p = torch.stack([0.1 * t, -0.2 * t, 0.05 * t], dim=-1)
        u = torch.stack([
            0.01 * torch.ones_like(t),
            -0.02 * torch.ones_like(t),
            0.005 * torch.ones_like(t),
            torch.ones_like(t),
            0.5 * torch.ones_like(t),
            -0.1 * torch.ones_like(t),
        ], dim=-1)
        coeff = tuple(c.float() for c in Interpolation.get_cubic_hermite_coeffs(t, u))
        self._imu = {
            "t_imu": t,
            "u_imu": u,
            "w_bias": torch.zeros(T, 3),
            "a_bias": torch.zeros(T, 3),
            "coeff": coeff,
            "t_start": 1.0e18,
        }
        self._gt = {
            "t_gt": t,
            "R_gt": R,
            "v_gt": v,
            "p_gt": p,
        }

    def load_imu(self, _):
        return self._imu

    def load_gt(self, _):
        return self._gt

    def get_start_timestamp(self, _):
        return self._imu["t_start"]


def test_gt_batch_shapes():
    dataset = FakeEurocDataset(mode="train", batch_size=4)
    batch = get_gt_se23_supervision_batch(dataset, 0, window_steps=20, batch_size=4)
    assert batch.mode == "gt_se23"
    assert batch.X_state.shape == (20, 4, 5, 5)
    assert batch.X_ref.shape == (20, 4, 5, 5)
    assert batch.supervised_indices.shape == (20,)


def test_vo_sampling_shapes():
    dataset = FakeEurocDataset(mode="train", batch_size=3)
    gt = dataset.load_gt(0)
    vo_traj = PoseTrajectory(t=gt["t_gt"][::10], R=gt["R_gt"][::10], p=gt["p_gt"][::10])
    batch = get_vo_pose_supervision_batch(
        dataset,
        0,
        vo_traj,
        window_steps=6,
        sup_stride=10,
        batch_size=3,
    )
    assert batch.mode == "vo_pose"
    assert batch.X_state.shape == (51, 3, 5, 5)
    assert batch.R_ref.shape[-2:] == (3, 3)
    assert batch.p_ref.shape[-1] == 3
    assert batch.supervised_indices[-1].item() == 50


def test_vo_rate_gt_uses_timing_only():
    dataset = FakeEurocDataset(mode="train", batch_size=3)
    gt = dataset.load_gt(0)
    vo_t = gt["t_gt"][::10]
    timing_a = PoseTrajectory(
        t=vo_t,
        R=gt["R_gt"][::10],
        p=gt["p_gt"][::10],
    )
    timing_b = PoseTrajectory(
        t=vo_t,
        R=Lie.SO3exp(torch.full((vo_t.numel(), 3), 0.7)),
        p=torch.full((vo_t.numel(), 3), 1.0e6),
    )
    torch.manual_seed(19)
    batch_a = get_vo_rate_gt_se23_supervision_batch(
        dataset,
        0,
        timing_a,
        window_steps=6,
        sup_stride=10,
        batch_size=3,
    )
    torch.manual_seed(19)
    batch_b = get_vo_rate_gt_se23_supervision_batch(
        dataset,
        0,
        timing_b,
        window_steps=6,
        sup_stride=10,
        batch_size=3,
    )
    assert batch_a.mode == "vo_rate_gt_se23"
    assert batch_a.X_state.shape == (51, 3, 5, 5)
    assert batch_a.X_ref.shape == (6, 3, 5, 5)
    assert torch.equal(batch_a.supervised_indices, torch.arange(0, 51, 10))
    assert torch.equal(
        build_gt_nll_supervised_indices(batch_a.supervised_indices),
        batch_a.supervised_indices,
    )
    assert torch.equal(batch_a.starts, batch_b.starts)
    assert torch.equal(batch_a.t_ref, batch_b.t_ref)
    assert torch.equal(batch_a.X_ref, batch_b.X_ref)


def test_vo_alignment_and_interpolation():
    dataset = FakeEurocDataset(mode="train", batch_size=2)
    gt = dataset.load_gt(0)
    R_offset = Lie.SO3exp(torch.tensor([0.1, -0.05, 0.02]))
    p_offset = torch.tensor([1.0, -0.2, 0.4])
    R_vo = torch.matmul(R_offset.transpose(-1, -2).unsqueeze(0), gt["R_gt"])
    p_vo = torch.matmul(
        R_offset.transpose(-1, -2),
        (gt["p_gt"] - p_offset).unsqueeze(-1),
    ).squeeze(-1)
    vo_traj = PoseTrajectory(t=gt["t_gt"], R=R_vo, p=p_vo)
    aligned = align_vo_first_gt(vo_traj, PoseTrajectory(t=gt["t_gt"], R=gt["R_gt"], p=gt["p_gt"]))
    R_eval, p_eval = interpolate_pose(gt["t_gt"][5:10], aligned)
    assert (p_eval - gt["p_gt"][5:10]).abs().max() < 1e-5
    assert Lie.SO3log(torch.matmul(gt["R_gt"][5:10].transpose(-1, -2), R_eval)).abs().max() < 1e-5


def test_stride_helpers():
    assert resolve_supervision_stride(0.005, sup_rate_hz=5.0) == 40
    assert build_supervised_indices(400, 40)[-1].item() == 399
    dense = torch.arange(64, dtype=torch.long)
    nll_indices = build_gt_nll_supervised_indices(dense, dense_stride=2)
    assert torch.equal(nll_indices, torch.arange(0, 64, 2, dtype=torch.long))
    assert nll_indices[-1].item() == 62
    assert torch.all(nll_indices[1:] - nll_indices[:-1] == 2)
    already_sparse = torch.tensor([0, 3, 6, 9], dtype=torch.long)
    assert torch.equal(
        build_gt_nll_supervised_indices(already_sparse, dense_stride=2),
        already_sparse,
    )


def test_cubic_hermite_searchsorted_matches_legacy_scan():
    """Check values, derivatives, knots, extrapolation, and query gradients."""
    dtype = torch.float64
    times = torch.tensor([0.0, 0.1, 0.4, 1.0, 1.7], dtype=dtype)
    data = torch.tensor(
        [
            [0.2, -0.3],
            [0.5, 0.4],
            [-0.2, 0.8],
            [1.1, -0.7],
            [0.3, 0.2],
        ],
        dtype=dtype,
    )
    coeff = Interpolation.get_cubic_hermite_coeffs(times, data)
    spline = Interpolation.CubicHermiteSpline(times, coeff)
    max_index = coeff[1].shape[-2] - 1

    def legacy_interpret(query: torch.Tensor):
        if query.dim() == 0:
            index = (query > times).sum() - 1
        else:
            index = (query.unsqueeze(-1) >= times).sum(dim=-1) - 1
        index = index.clamp(0, max_index)
        return query - times[index], index

    def legacy_evaluate(query: torch.Tensor):
        fractional, index = legacy_interpret(query)
        fractional = fractional / (times[index + 1] - times[index])
        if query.dim() != 0:
            fractional = fractional.unsqueeze(-1)
        a, b, c, d = coeff
        inner = c[..., index, :] + d[..., index, :] * fractional
        inner = b[..., index, :] + inner * fractional
        return a[..., index, :] + inner * fractional

    def legacy_derivative(query: torch.Tensor):
        fractional, index = legacy_interpret(query)
        interval = times[index + 1] - times[index]
        fractional = fractional / interval
        if query.dim() != 0:
            fractional = fractional.unsqueeze(-1)
            interval = interval.unsqueeze(-1)
        _, b, c, d = coeff
        inner = 3 * d[..., index, :] * fractional + 2 * c[..., index, :]
        return (b[..., index, :] + inner * fractional) / interval

    queries = torch.tensor(
        [-0.2, 0.0, 0.03, 0.1, 0.4, 0.99, 1.0, 1.7, 2.0],
        dtype=dtype,
        requires_grad=True,
    )
    expected_fractional, expected_index = legacy_interpret(queries)
    actual_fractional, actual_index = spline._interpret_t(queries)
    assert torch.equal(actual_index, expected_index)
    assert torch.equal(actual_fractional, expected_fractional)
    assert torch.equal(spline.evaluate(queries), legacy_evaluate(queries))
    assert torch.equal(spline.derivative(queries), legacy_derivative(queries))

    actual_objective = (
        spline.evaluate(queries) + 0.1 * spline.derivative(queries)
    ).sum()
    actual_gradient = torch.autograd.grad(actual_objective, queries)[0]
    reference_queries = queries.detach().clone().requires_grad_(True)
    reference_objective = (
        legacy_evaluate(reference_queries)
        + 0.1 * legacy_derivative(reference_queries)
    ).sum()
    reference_gradient = torch.autograd.grad(
        reference_objective,
        reference_queries,
    )[0]
    assert torch.allclose(
        actual_gradient,
        reference_gradient,
        atol=2e-14,
        rtol=2e-13,
    ), (
        float((actual_gradient - reference_gradient).abs().max()),
        actual_gradient,
        reference_gradient,
    )

    for scalar_value in queries.detach():
        expected_fractional, expected_index = legacy_interpret(scalar_value)
        actual_fractional, actual_index = spline._interpret_t(scalar_value)
        assert torch.equal(actual_index, expected_index)
        assert torch.equal(actual_fractional, expected_fractional)
        assert torch.equal(
            spline.evaluate(scalar_value),
            legacy_evaluate(scalar_value),
        )
        assert torch.equal(
            spline.derivative(scalar_value),
            legacy_derivative(scalar_value),
        )


def main():
    test_gt_batch_shapes()
    test_vo_sampling_shapes()
    test_vo_rate_gt_uses_timing_only()
    test_vo_alignment_and_interpolation()
    test_stride_helpers()
    test_cubic_hermite_searchsorted_matches_legacy_scan()
    print("PLATO supervision smoke tests passed")


if __name__ == "__main__":
    main()
