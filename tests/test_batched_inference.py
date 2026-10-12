import copy

import pytest
import torch

from network_for_BiadDy import VFSE23vbiasBase, bw_func_choose, ba_func_choose
from plato.rollout import integrate_batch_paper_zoh


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_padded_production_rollout_matches_independent_serial_prefixes(device):
    torch.manual_seed(22)
    model = VFSE23vbiasBase(biasfunc_w=bw_func_choose("bw_func_net", device), biasfunc_a=ba_func_choose("ba_func_net_learnable_rate_tanh", device), device=device).double()
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    t = torch.arange(12, device=device, dtype=torch.float64)*0.001
    y0 = torch.zeros(2,16, device=device, dtype=torch.float64)
    R0 = torch.eye(3, device=device, dtype=torch.float64).repeat(2,1,1)
    u = torch.randn(11,2,6, device=device, dtype=torch.float64)
    du = torch.randn_like(u)
    lengths = torch.tensor([8,12],device=device)
    with torch.no_grad():
        actual_y, actual_R = integrate_batch_paper_zoh(model, y0, R0, t, valid_lengths=lengths, sampled_imu=(u,du))
        for row, count in enumerate(lengths.tolist()):
            independent = copy.deepcopy(model)
            expected_y, expected_R = integrate_batch_paper_zoh(independent, y0[row:row+1], R0[row:row+1], t[:count], sampled_imu=(u[:count-1,row:row+1],du[:count-1,row:row+1]))
            torch.testing.assert_close(actual_y[:count,row],expected_y[:,0],rtol=1e-10,atol=1e-12)
            torch.testing.assert_close(actual_R[:count,row],expected_R[:,0],rtol=1e-10,atol=1e-12)
        torch.testing.assert_close(actual_y[8:,0,3:],actual_y[7,0,3:].expand(4,13))
        torch.testing.assert_close(actual_R[8:,0],actual_R[7,0].expand(4,3,3))


@pytest.mark.parametrize("bias_init", ["first", "zero"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_native_inference_matches_original_test_with_mixed_time_precision(device, bias_init, tmp_path):
    import Interpolation
    from dataset import BaseDataset, pload
    from learning import Test
    from plato.inference import rollout_native_sequences
    from plato.supervision_smoke_test import FakeEurocDataset

    class Dataset:
        get_full_trajectory = BaseDataset.get_full_trajectory
        get_coeff = BaseDataset.get_coeff
        get_w_bias = BaseDataset.get_w_bias
        get_a_bias = BaseDataset.get_a_bias
        sequences = ["short", "long"]
        dt = 0.0010018348693847656
        gravity = torch.tensor([0., 0., -9.7879])

        def __init__(self):
            self.records = []
            for row, count in enumerate((31, 57)):
                fixture = FakeEurocDataset()
                imu, gt = fixture._imu, fixture._gt
                for record in (imu, gt):
                    for key, value in record.items():
                        if isinstance(value, torch.Tensor):
                            record[key] = value[:count].clone()
                t = torch.arange(count, dtype=torch.float64)*self.dt
                imu["t_imu"], gt["t_gt"] = t, t.float()
                imu["u_imu"] = torch.randn(count, 6)*0.1
                imu["u_imu"][:,5] += 9.7879
                imu["w_bias"][:] = 0.01*(row+1)
                imu["a_bias"][:] = 0.1*(row+1)
                imu["coeff"] = tuple(value.float() for value in Interpolation.get_cubic_hermite_coeffs(t, imu["u_imu"]))
                self.records.append((imu,gt))

        def load_imu(self, index):
            return self.records[index][0]

        def load_gt(self, index):
            return self.records[index][1]

        def length(self):
            return len(self.records)

        def get_start_timestamp(self, index):
            return self.records[index][0]["t_start"]

    torch.manual_seed(75)
    dataset = Dataset()
    model = VFSE23vbiasBase(biasfunc_w=bw_func_choose("bw_func_relaxation",device), biasfunc_a=ba_func_choose("ba_func_relaxation",device),device=device)
    # Exercise learned nonzero targets rather than only the zero initialization.
    with torch.no_grad():
        model.biasfunc_w.target_net[-1].weight.normal_(std=0.02)
        model.biasfunc_a.target_net[-1].weight.normal_(std=0.02)
        model.set_gravity(dataset.gravity.to(device))
        actual = rollout_native_sequences(dataset,copy.deepcopy(model),device,bias_init)
        Test(dataset,str(tmp_path),copy.deepcopy(model.biasfunc_w),copy.deepcopy(model.biasfunc_a),device=device,recompute=True,bias_init_mode=bias_init,compute_raw_int=False,mean_discretization="paper_zoh")
    for sequence, prediction in zip(dataset.sequences,actual):
        expected = pload(str(tmp_path / "Sequnces-results" / sequence / "FullTra_result.p"))
        for key in ("sol_pred", "R_pred", "net_imu"):
            torch.testing.assert_close(prediction[key],expected[key],rtol=2e-5,atol=2e-6)
        assert torch.equal(prediction["t_gt"],expected["t_gt"])
