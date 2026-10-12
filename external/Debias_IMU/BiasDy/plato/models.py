"""Contractive neural bias dynamics with serialized physical scales."""
from __future__ import annotations
import torch
from torch import nn


class BiasRelaxation(nn.Module):
    """db/dt = rate * (bounded_target_theta(raw_imu) - b).

    The target is independent of every state coordinate: df/db=-rate I.
    Euler updates are convex combinations when rate*dt<=1. Targets retain
    gyro rad/s or accel m/s^2 units; rate is in 1/s. All physical scales are
    serialized buffers. This is an explicitly different stable NODE variant.
    """
    def __init__(self, bias_offset: int, target_limit: float, rate: float = 10.0):
        super().__init__()
        if bias_offset not in (4,13) or target_limit <= 0 or rate <= 0:
            raise ValueError("Expected a valid bias slice and positive target bound/rate")
        self.bias_offset = bias_offset
        self.register_buffer("target_limit",torch.tensor(float(target_limit)))
        self.register_buffer("relaxation_rate",torch.tensor(float(rate)))
        self.register_buffer("input_scale",torch.tensor([1.,1.,1.,9.81,9.81,9.81]))
        self.target_net = nn.Sequential(nn.Linear(6,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,3))
        nn.init.zeros_(self.target_net[-1].weight)
        nn.init.zeros_(self.target_net[-1].bias)

    def forward(self,y,imu_meas,imu_meas_dot,R0):
        target = self.target_limit*torch.tanh(self.target_net(imu_meas/self.input_scale))
        return self.relaxation_rate*(target-y[...,self.bias_offset:self.bias_offset+3])
