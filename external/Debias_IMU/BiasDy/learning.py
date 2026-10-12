"""Serial evaluation using the same native ZOH rollout as PLATO training."""
import os
import torch
import Interpolation
import lie_algebra as Lie
from dataset import BaseDataset, EUROCDataset, pload, pdump
from network_for_BiadDy import bw_func_net, ba_func_net, VFSE23vbiasBase
from plato.rollout import integrate_batch_paper_zoh
from utils import bmv


def _dataset_gravity(dataset: BaseDataset, device="cpu", dtype=torch.float32):
    gravity = getattr(dataset, "gravity", None)
    if gravity is None:
        gravity = getattr(dataset, "g_const", None)
    if gravity is None:
        gravity = [0, 0, -9.81]
    return torch.as_tensor(gravity, device=device, dtype=dtype)


def Test(
    dataset_test: EUROCDataset,
    output_dir: str,
    bw_model: bw_func_net,
    ba_model: ba_func_net,
    integral_method="euler",
    device="cpu",
    recompute=False,
    bias_init_mode="first",
    bias_init_seconds=5.0,
    compute_raw_int=True,
    disable_acc_bias=False,
    mean_discretization="paper_zoh",
):
    if mean_discretization != "paper_zoh":
        raise ValueError("The release supports mean_discretization=paper_zoh")
    if mean_discretization == "paper_zoh" and integral_method != "euler":
        raise ValueError(
            "paper_zoh uses Euler bias dynamics and requires integral_method='euler'"
        )

    def calculate_raw_imu(dataset: EUROCDataset, index, device, path: str):
        if not os.path.exists(path):
            print("No results using raw data, calculate it now!")
            t_imu, u_imu, X_gt, t_gt = dataset.get_full_trajectory(index)
            u_imu = u_imu.to(device)
            X_gt = X_gt.to(device)
            t_gt = t_gt.to(device)
            X_raw = torch.zeros_like(X_gt)
            X_raw[0] = X_gt[0]
            g_const = _dataset_gravity(dataset, device=device, dtype=X_gt.dtype)
            for i in range(1, X_gt.shape[0]):
                dt = t_gt[i] - t_gt[i-1]
                X_raw[i,:3,:3] = X_raw[i-1,:3,:3] @ Lie.SO3exp(u_imu[i-1,:3] * dt)
                X_raw[i,:3,3] = X_raw[i-1,:3,3] + (bmv(X_raw[i-1,:3,:3], u_imu[i-1,3:]) +g_const )* dt
                X_raw[i,:3,4] = X_raw[i-1,:3,4] + X_raw[i-1,:3,3] * dt
            datadic = {'X_raw': X_raw}
            pdump(datadic, path)
        else:
            datadic = pload(path)
            X_raw = datadic['X_raw']

    model = VFSE23vbiasBase(biasfunc_w=bw_model.to(device), biasfunc_a=ba_model.to(device), device=device)
    model.set_gravity(_dataset_gravity(dataset_test, device=device))
    model.set_bw_zero(False)
    model.set_ba_zero(disable_acc_bias)
    model.eval()

    for index in range(dataset_test.length()):
        path_tmp = os.path.join(output_dir, "Sequnces-results", dataset_test.sequences[index])
        os.makedirs(path_tmp, exist_ok=True)
        result_path = os.path.join(path_tmp, "FullTra_result.p")
        if not recompute and os.path.exists(result_path):
            cached_result = pload(result_path)
            cached_discretization = cached_result.get(
                "mean_discretization", "legacy_euler"
            )
            if cached_discretization == mean_discretization:
                continue
        print("Testing trajectory: ", dataset_test.sequences[index])
        # read imu data
        coeff, time = dataset_test.get_coeff(index)
        Spline = Interpolation.CubicHermiteSpline(time, coeff, device)

        # read initial condition
        _, u_imu, X_gt, t_gt = dataset_test.get_full_trajectory(index,device=device) # shape [N, 5, 5], [N]
        R0 = X_gt[0,:3,:3].clone()
        bias_w_gt,_ = dataset_test.get_w_bias(index)
        bias_a_gt,_ = dataset_test.get_a_bias(index)
        if bias_init_mode == "zero":
            bw0 = torch.zeros(3, device=device, dtype=X_gt.dtype)
            ba0 = torch.zeros(3, device=device, dtype=X_gt.dtype)
        elif bias_init_mode == "first":
            bw0 = bias_w_gt[0].to(device)
            ba0 = bias_a_gt[0].to(device)
        elif bias_init_mode == "mean":
            _, t_bias = dataset_test.get_w_bias(index)
            mask = t_bias <= t_bias[0] + float(bias_init_seconds)
            if not torch.any(mask):
                mask = torch.zeros_like(t_bias, dtype=torch.bool)
                mask[0] = True
            bw0 = bias_w_gt[mask].mean(dim=0).to(device)
            ba0 = bias_a_gt[mask].mean(dim=0).to(device)
        else:
            raise ValueError("bias_init_mode must be 'zero', 'first' or 'mean'")
        Vec3zeros = torch.zeros(3, device=device)
        # y0 = torch.cat([Vec3zeros, t_gt[0].unsqueeze(-1), bw0, X_gt[0,:3,3], X_gt[0,:3,4], ba0, Vec3zeros], dim=-1).unsqueeze(0) # [16,1]
        y0 = torch.cat([Vec3zeros, t_gt[0].unsqueeze(-1), bw0, X_gt[0,:3,3], X_gt[0,:3,4], ba0], dim=-1).unsqueeze(0) # [16,1]

        # integrate
        model.set_u_func(Spline.evaluate)
        model.set_u_dot_func(Spline.derivative)
        model.set_R0(R0) # don't forget to set R0
        if hasattr(model, "set_imu_window_context"):
            model.set_imu_window_context(
                float(dataset_test.dt),
                t_min=time[0],
                t_max=time[-1],
            )
        if model.biasfunc_w is not None and hasattr(model.biasfunc_w, "set_initial_ba"):
            model.biasfunc_w.set_initial_ba(y0[..., 13:16])
        if model.biasfunc_a is not None and hasattr(model.biasfunc_a, "set_initial_ba"):
            model.biasfunc_a.set_initial_ba(y0[..., 13:16])
        sol_pred, R_pred = integrate_batch_paper_zoh(
            model, y0, R0.unsqueeze(0), t_gt
        )
        sol_pred = sol_pred.squeeze(1) # oder: [xi, t, bw, v, p, ba]
        R_pred = R_pred.squeeze(1)
        net_imu = u_imu - torch.cat([sol_pred[:,4:7], sol_pred[:,13:16]], dim=-1)
        datadic = {'sol_pred': sol_pred, 'R_pred': R_pred, 'X_gt': X_gt, 't_gt': t_gt,'net_imu':net_imu, 'imu_raw': u_imu, 'timestamp_start': dataset_test.get_start_timestamp(index), 'mean_discretization': mean_discretization}
        pdump(datadic, path_tmp, "FullTra_result.p")

    if compute_raw_int:
        for index in range(dataset_test.length()):
            # calculate raw imu data integration
            path_raw_int = os.path.join(output_dir, "raw_int", dataset_test.sequences[index] + ".p")
            os.makedirs(os.path.dirname(path_raw_int), exist_ok=True)
            calculate_raw_imu(dataset_test, index, device, path_raw_int)
