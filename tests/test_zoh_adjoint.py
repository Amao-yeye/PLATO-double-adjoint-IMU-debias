"""Finite-step invariance and analytic adjoint tests using production kernels."""
import pytest
import torch

import lie_algebra as Lie
from plato.invariant_discretization import so3_gamma_matrices, so3_gamma_action_jacobians, se23_zoh_step
from plato.analytic_gradient import invariant_transition_matrices, adjoint_bias_gradient


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("angle", [0.0, 1e-6, 0.04, 0.2])
def test_gamma_action_jacobians(dtype, angle):
    phi = torch.tensor([angle, -0.3 * angle, 0.4 * angle], dtype=dtype)
    vector = torch.tensor([0.8, -0.1, 9.4], dtype=dtype)
    analytic = so3_gamma_action_jacobians(phi, vector)
    for index in range(2):
        reference = torch.autograd.functional.jacobian(lambda x: so3_gamma_matrices(x)[index] @ vector, phi)
        torch.testing.assert_close(analytic[index], reference, rtol=3e-4 if dtype == torch.float32 else 1e-9, atol=2e-5 if dtype == torch.float32 else 1e-10)
        assert torch.isfinite(reference).all()


def test_multi_step_outer_adjoint_matches_autograd():
    torch.manual_seed(123)
    dtype = torch.float64
    steps, batch = 8, 2
    dt = torch.linspace(0.002, 0.008, steps, dtype=dtype)
    bias = (0.01 * torch.randn(steps, batch, 6, dtype=dtype)).requires_grad_()
    omega = 0.8 * torch.randn(steps, batch, 3, dtype=dtype)
    force = torch.randn(steps, batch, 3, dtype=dtype) + torch.tensor([0., 0., 9.81])
    R = Lie.SO3exp(0.2 * torch.randn(batch, 3, dtype=dtype))
    v, p = torch.randn(batch, 3, dtype=dtype), torch.randn(batch, 3, dtype=dtype)
    rotations, velocities, positions = [R], [v], [p]
    for k in range(steps):
        R, v, p, _ = se23_zoh_step(R, v, p, omega[k] - bias[k, :, 3:], force[k] - bias[k, :, :3], dt[k])
        rotations.append(R)
        velocities.append(v)
        positions.append(p)
    R, v, p = map(torch.stack, (rotations, velocities, positions))
    target_v, target_p = torch.randn_like(v), torch.randn_like(p)
    loss = 0.5 * ((v - target_v).square().sum() + (p - target_p).square().sum())
    reference = torch.autograd.grad(loss, bias)[0]
    # Local gradients in the left tangent [rotation, velocity, position].
    X = Lie.SEn3fromSO3Rn(R.detach(), torch.cat([v.detach(), p.detach()], dim=-1))
    delta = torch.zeros(steps + 1, batch, 9, dtype=dtype, requires_grad=True)
    perturbed = Lie.SEn3exp(delta) @ X
    local_loss = 0.5 * ((perturbed[..., :3, 3] - target_v).square().sum() + (perturbed[..., :3, 4] - target_p).square().sum())
    state_gradient = torch.autograd.grad(local_loss, delta)[0]
    F, B = invariant_transition_matrices(R.detach(), v.detach(), p.detach(), dt=dt, discretization="paper_zoh", corrected_omega=(omega - bias[..., 3:]).detach(), corrected_specific_force=(force - bias[..., :3]).detach())
    actual = adjoint_bias_gradient(state_gradient, F, B).bias_gradient
    torch.testing.assert_close(actual, reference, rtol=2e-8, atol=1e-10)
    eye = torch.eye(3, dtype=dtype).expand_as(R)
    torch.testing.assert_close(R.transpose(-1, -2) @ R, eye, rtol=1e-12, atol=1e-12)
