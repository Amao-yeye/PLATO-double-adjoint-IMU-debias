"""Compare the real nested training backward pass against full autograd."""
import copy
import json
from dataclasses import replace

import pytest
import torch

from dataset import BaseDataset
from network_for_BiadDy import VFSE23vbiasBase, ba_func_choose, bw_func_choose
from plato.nll import PlatoNoiseParams
from plato.supervision import get_gt_se23_supervision_batch, get_vo_pose_supervision_batch, PoseTrajectory
from plato.supervision_smoke_test import FakeEurocDataset
from plato.training import _integrate_batch, _nll_loss_for_batch, _train_nll_double_adjoint_step
from train_euroc_release import resolve_parameters, REPO


class SyntheticDataset(FakeEurocDataset):
    construct_init_forodeint = staticmethod(BaseDataset.construct_init_forodeint)


@pytest.mark.parametrize("supervision", ["gt_se23", "vo_pose"])
@pytest.mark.parametrize("architecture", ["baseline", "relaxation"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_nested_nll_double_adjoint_gradient(supervision, architecture, device, tmp_path):
    torch.manual_seed(321)
    dataset = SyntheticDataset(batch_size=2)
    for record in (dataset._imu, dataset._gt):
        for name, value in record.items():
            if isinstance(value, torch.Tensor):
                record[name] = value.double()
            elif isinstance(value, tuple):
                record[name] = tuple(item.double() for item in value)
    recipe = json.loads((REPO / "configs/euroc_release.json").read_text())
    _, config = resolve_parameters(recipe, tmp_path, tmp_path, device)
    config = replace(config, supervision=supervision, window_steps=8, batch_size=2, covariance_gradient_mode="autograd")
    if supervision == "gt_se23":
        batch = get_gt_se23_supervision_batch(dataset, 0, window_steps=8, batch_size=2, device=device)
    else:
        trajectory = PoseTrajectory(t=dataset._gt["t_gt"][::2], R=dataset._gt["R_gt"][::2], p=dataset._gt["p_gt"][::2])
        batch = get_vo_pose_supervision_batch(dataset, 0, trajectory, window_steps=8, sup_stride=2, batch_size=2, device=device)
    bg_type = "bw_func_relaxation" if architecture == "relaxation" else "bw_func_net"
    ba_type = "ba_func_relaxation" if architecture == "relaxation" else "ba_func_net_learnable_rate_tanh"
    model = VFSE23vbiasBase(biasfunc_w=bw_func_choose(bg_type,device),biasfunc_a=ba_func_choose(ba_type,device),device=device).double().to(device)
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    reference_model = copy.deepcopy(model)
    noise = PlatoNoiseParams(init_sigma_a=0.1, init_sigma_g=0.1, isotropic=True, device=device, dtype=torch.float64)
    reference_noise = copy.deepcopy(noise)
    solution, R = _integrate_batch(reference_model, dataset, batch, config)
    reference_loss, _ = _nll_loss_for_batch(batch, solution, R, config, reference_noise)
    reference_loss.backward()
    actual_config = replace(config, gradient_mode="double_adjoint", covariance_gradient_mode="forward_sensitivity")
    actual_loss, _, _ = _train_nll_double_adjoint_step(model, dataset, batch, actual_config, noise)
    torch.testing.assert_close(actual_loss, reference_loss, rtol=1e-10, atol=1e-7)
    for name in ("biasfunc_w", "biasfunc_a"):
        actual_parameters = list(getattr(model, name).parameters())
        reference_parameters = list(getattr(reference_model, name).parameters())
        actual = torch.cat([parameter.grad.flatten() for parameter in actual_parameters])
        expected = torch.cat([parameter.grad.flatten() for parameter in reference_parameters])
        relative_error = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
        assert relative_error < 2e-4, (supervision, device, name, relative_error.item())
    for actual, expected in zip(noise.parameters(), reference_noise.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-8, atol=1e-6)
