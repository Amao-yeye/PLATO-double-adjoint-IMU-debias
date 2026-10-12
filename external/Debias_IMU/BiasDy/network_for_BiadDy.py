"""Bias networks supported by the EuRoC/AquaLux release and gradient checks."""
import torch
from torch import nn
import lie_algebra as Lie
from utils import bmv


class bw_func_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 9),
        )
        self.net2 = nn.Sequential(
            nn.Linear(9, 512),
            nn.Tanh(),
            nn.Linear(512, 9),
        )
        self.linear2 = nn.Linear(9, 3)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, y, imu_meas, imu_meas_dot, R0):
        """y:shape (batch_size, 16) where 16 = xi, t, bw, v, p, ba
            imu_meas: shape (batch_size, 6) where 6 = w, a
            imu_meas_dot: shape (batch_size, 6) where 6 = w_dot, a_dot"""
        x = torch.cat([ y[..., 4:7], imu_meas[...,:3], imu_meas_dot[...,:3]], dim=-1) # order:  bw, w, w_dot
        x = self.net(x) + x
        x = self.net2(x) + x
        return self.linear2(x)


class ba_func_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 9),
        )
        self.net2 = nn.Sequential(
            nn.Linear(9, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 9),
        )
        self.linear2 = nn.Linear(9, 3)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)
    def forward(self, y, imu_meas, imu_meas_dot, R0):
        """y:shape (batch_size, 16) where 16 = xi, t, bw, v, p, ba
            imu_meas: shape (batch_size, 6) where 6 = w, a
            imu_meas_dot: shape (batch_size, 6) where 6 = w_dot, a_dot"""
        x = torch.cat([y[..., 13:16], imu_meas[...,3:],imu_meas_dot[...,3:]], dim=-1) # order:  ba, a, a_dot
        x = self.net(x) + x
        x = self.net2(x) + x
        return self.linear2(x)


class ba_func_net_learnable_rate_tanh(ba_func_net):
    """BA NODE with a smooth, learnable rate limit applied component-wise.

    The same parameterization is used during MSE and NLL. The positive
    ``rate_unconstrained`` coordinate is a regular model parameter saved in
    every checkpoint. Training may use projected continuation to shrink its
    feasible upper bound without changing the network or loss definition.
    """

    rate_floor = 1e-4
    # Begin in the near-identity regime so short-window MSE can discover the
    # unconstrained solution. Projected continuation later exposes the same
    # model to the long-rollout-stable 5e-3 feasible set.
    initial_rate = 1e4

    @staticmethod
    def _softplus_inverse(positive_value):
        # x + log(1 - exp(-x)) is stable both near zero and for x >> 1.
        return positive_value + torch.log(-torch.expm1(-positive_value))

    def __init__(self):
        super().__init__()
        positive_rate = torch.tensor(
            self.initial_rate - self.rate_floor, dtype=torch.float32
        )
        self.rate_unconstrained = nn.Parameter(
            self._softplus_inverse(positive_rate)
        )

    def rate_limit(self):
        return self.rate_floor + torch.nn.functional.softplus(
            self.rate_unconstrained
        )

    def project_rate_limit(self, maximum: float):
        if maximum <= self.rate_floor:
            raise ValueError("BA rate maximum must exceed rate_floor")
        with torch.no_grad():
            if self.rate_limit() <= maximum:
                return
            positive_rate = torch.as_tensor(
                maximum - self.rate_floor,
                device=self.rate_unconstrained.device,
                dtype=self.rate_unconstrained.dtype,
            )
            self.rate_unconstrained.copy_(
                self._softplus_inverse(positive_rate)
            )

    def forward(self, y, imu_meas, imu_meas_dot, R0):
        raw_rate = super().forward(y, imu_meas, imu_meas_dot, R0)
        rate_limit = self.rate_limit().to(
            device=raw_rate.device, dtype=raw_rate.dtype
        )
        return rate_limit * torch.tanh(raw_rate / rate_limit)


class VFSE23vbiasBase(torch.nn.Module):
    def __init__(self, u_func = None, u_dot_func = None,\
                biasfunc_w: bw_func_net = None, biasfunc_a: ba_func_net = None, biasfunc_v = None, \
                R0 = torch.eye(3), device = "cuda", gravity = None):
        super().__init__()

        # biasfunc_w, biasfunc_a, biasfunc_v should be a torch.nn.Module
        assert isinstance(biasfunc_w, torch.nn.Module) or biasfunc_w is None
        assert isinstance(biasfunc_a, torch.nn.Module) or biasfunc_a is None

        self.biasfunc_w = biasfunc_w
        self.biasfunc_a = biasfunc_a

        self.u_func = u_func
        self.u_dot_func = u_dot_func
        self._device = device
        self.R0 = R0.to(device)
        if gravity is None:
            gravity = [0, 0, -9.81]
        self.set_gravity(gravity)

        self._set_bw_zero = False
        self._set_ba_zero = False
        self._imu_window_dt = 0.001
        self._imu_window_t_min = None
        self._imu_window_t_max = None

        gyro_Rot = 0.05*torch.randn(3, 3).to(device)
        self.gyro_Rot = torch.nn.Parameter(gyro_Rot)

    def state_dict(self, *args, **kwargs):
        original_dict = super().state_dict(*args, **kwargs)
        keys_to_remove = [key for key in original_dict.keys() if key.startswith('biasfunc')]
        for key in keys_to_remove:
            original_dict.pop(key)
        return original_dict

    def set_u_func(self, u_func):
        self.u_func = u_func
    def set_u_dot_func(self, u_dot_func):
        self.u_dot_func = u_dot_func

    def set_imu_window_context(self, dt, t_min=None, t_max=None):
        self._imu_window_dt = float(dt)
        self._imu_window_t_min = t_min
        self._imu_window_t_max = t_max

    def _evaluate_acc_window(self, t_abs, steps):
        offsets = -self._imu_window_dt * torch.arange(
            int(steps) - 1, -1, -1, device=t_abs.device, dtype=t_abs.dtype
        )
        t_window = t_abs.unsqueeze(-1) + offsets
        if self._imu_window_t_min is not None:
            t_min = torch.as_tensor(self._imu_window_t_min, device=t_abs.device, dtype=t_abs.dtype)
            t_window = torch.maximum(t_window, t_min)
        if self._imu_window_t_max is not None:
            t_max = torch.as_tensor(self._imu_window_t_max, device=t_abs.device, dtype=t_abs.dtype)
            t_window = torch.minimum(t_window, t_max)
        u_window = self.u_func(t_window.reshape(-1))
        u_window = u_window.to(device=t_abs.device, dtype=t_abs.dtype)
        return u_window.reshape(*t_abs.shape, int(steps), -1)[..., 3:]

    def callback_change_chart(self, R0):
        self.set_R0(R0)
    def set_R0(self, R0):
        self.R0 = R0.clone().to(self._device)

    def set_gravity(self, gravity):
        self.g_const = torch.as_tensor(gravity, dtype=torch.float32, device=self._device)

    def __call__(self, t, y: torch.Tensor):
        """y:shape (batch_size, 16) where 16 = xi, t, bw, v, p, ba"""
        #################### !!!!Important!!! ######################
        ##  remenber to assign self.R0 before using this function ##
        ############################################################
        t_abs = y[..., 3] # shape (batch_size,)
        imu_meas = self.u_func(t_abs)
        imu_meas = imu_meas.to(device=y.device, dtype=y.dtype)
        w_tilde = imu_meas[..., :3]
        imu_meas_dot = self.u_dot_func(t_abs)
        imu_meas_dot = imu_meas_dot.to(device=y.device, dtype=y.dtype)
        if self._set_bw_zero:
            bw = torch.zeros_like(y[..., 4:7])
            bw_dot = torch.zeros_like(y[..., 4:7])
        else:
            bw = y[..., 4:7]
            bw_dot = self.biasfunc_w(y, imu_meas, imu_meas_dot, self.R0) #* self.alpha
        if self._set_ba_zero:
            ba = torch.zeros_like(y[..., 13:16])
            ba_dot = torch.zeros_like(y[..., 13:16])
        else:
            ba = y[..., 13:16]
            if getattr(self.biasfunc_a, "uses_acc_window", False):
                acc_window = self._evaluate_acc_window(
                    t_abs, getattr(self.biasfunc_a, "acc_window_steps", 1)
                )
                ba_dot = self.biasfunc_a(y, imu_meas, imu_meas_dot, self.R0, acc_window)
            else:
                ba_dot = self.biasfunc_a(y, imu_meas, imu_meas_dot, self.R0)
        xi_dot = bmv(Lie.SO3rightJacoInv(y[..., :3]),w_tilde - bw ) # shape (batch_size, 3)
        t_dot = torch.ones(*t_abs.shape, 1).to(y.device)  # shape (batch_size, 1)
        Rt = self.R0.to(y.device) @ Lie.SO3exp(y[..., :3])
        g_const = self.g_const.to(device=y.device, dtype=y.dtype)
        v_dot = bmv(Rt, imu_meas[...,3:] - ba) + g_const.expand_as(y[..., 13:16]) # shape (batch_size, 3)
        p_dot = y[..., 7:10]
        return torch.cat([xi_dot, t_dot, bw_dot, v_dot, p_dot, ba_dot], dim=-1)

    def set_bw_zero(self, flag):
        self._set_bw_zero = flag

    def set_ba_zero(self, flag):
        self._set_ba_zero = flag

    def freeze_func_bw(self):
        if self.biasfunc_w is not None:
            for param in self.biasfunc_w.parameters():
                param.requires_grad = False

    def freeze_func_ba(self):
        if self.biasfunc_a is not None:
            for param in self.biasfunc_a.parameters():
                param.requires_grad = False

    def unfreeze_func_bw(self):
        if self.biasfunc_w is not None:
            for param in self.biasfunc_w.parameters():
                param.requires_grad = True

    def unfreeze_func_ba(self):
        if self.biasfunc_a is not None:
            for param in self.biasfunc_a.parameters():
                param.requires_grad = True


def bw_func_choose(network_type, device="cuda"):
    if network_type == "bw_func_net":
        return bw_func_net().to(device)
    if network_type == "bw_func_relaxation":
        from plato.models import BiasRelaxation
        return BiasRelaxation(bias_offset=4, target_limit=0.05).to(device)
    raise ValueError(f"Unsupported gyro network: {network_type}")


def ba_func_choose(network_type, device="cuda"):
    if network_type == "ba_func_net":
        return ba_func_net().to(device)
    if network_type == "ba_func_net_learnable_rate_tanh":
        return ba_func_net_learnable_rate_tanh().to(device)
    if network_type == "ba_func_relaxation":
        from plato.models import BiasRelaxation
        return BiasRelaxation(bias_offset=13, target_limit=1.0).to(device)
    raise ValueError(f"Unsupported accelerometer network: {network_type}")
