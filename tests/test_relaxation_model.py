import pytest
import torch
from plato.models import BiasRelaxation


@pytest.mark.parametrize("offset,limit",[(4,0.05),(13,1.0)])
def test_relaxation_jacobian_and_euler_bound(offset,limit):
    torch.manual_seed(5)
    model = BiasRelaxation(offset,limit).double()
    with torch.no_grad():
        for p in model.parameters(): p.uniform_(-1,1)
    y = torch.randn(16,dtype=torch.float64)
    imu = torch.randn(6,dtype=torch.float64)
    jac = torch.autograd.functional.jacobian(lambda state:model(state,imu,torch.zeros(6),torch.eye(3)),y)
    expected = torch.zeros(3,16,dtype=torch.float64)
    expected[:,offset:offset+3] = -10*torch.eye(3,dtype=torch.float64)
    torch.testing.assert_close(jac,expected)
    target = y[offset:offset+3]+model(y,imu,torch.zeros(6),torch.eye(3))/10
    assert target.abs().max() <= limit
    next_bias = y[offset:offset+3]+0.001*model(y,imu,torch.zeros(6),torch.eye(3))
    assert next_bias.abs().max() <= max(y[offset:offset+3].abs().max(),limit)
