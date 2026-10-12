from __future__ import annotations

import os
import sys

import torch
from torch.utils.tensorboard import SummaryWriter

BIASDY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEBIAS_ROOT = os.path.abspath(os.path.join(BIASDY_DIR, ".."))
if BIASDY_DIR not in sys.path:
    sys.path.insert(0, BIASDY_DIR)
if DEBIAS_ROOT not in sys.path:
    sys.path.insert(0, DEBIAS_ROOT)

import lie_algebra as Lie
from BiasDy_loss import SO3log_loss
from dataset import BaseDataset, pdump
from utils import bmv


def train_linear_baseline(
    dataset_train: BaseDataset,
    dataset_val: BaseDataset,
    outputdir: str,
    epochs: int = 5,
    device: str = "cuda",
    lr: float = 5e-3,
    weight_decay: float = 0.0,
    val_freq: int = 5,
    save_freq: int = 5,
) -> tuple[str, str]:
    gyro_path = train_linear_gyro(
        dataset_train, dataset_val, outputdir, epochs, device, lr, weight_decay, val_freq, save_freq
    )
    acc_path = train_linear_acc(
        dataset_train, dataset_val, outputdir, gyro_path, epochs, device, lr, weight_decay, val_freq, save_freq
    )
    return gyro_path, acc_path


def train_linear_gyro(
    dataset_train: BaseDataset,
    dataset_val: BaseDataset,
    outputdir: str,
    epochs: int,
    device: str,
    lr: float,
    weight_decay: float,
    val_freq: int,
    save_freq: int,
) -> str:
    torch.manual_seed(3407)
    A = torch.nn.Parameter((torch.eye(3) + 0.01 * torch.randn(3, 3)).to(device))
    b = torch.nn.Parameter(0.01 * torch.randn(3).to(device))
    optimizer = torch.optim.Adam([A, b], lr=lr, weight_decay=weight_decay)
    writer = SummaryWriter(os.path.join(outputdir, "GyroAxb_logs"))
    save_dir = os.path.join(outputdir, "GyroAxbweights")
    os.makedirs(save_dir, exist_ok=True)

    last_val = float("inf")
    for epoch in range(epochs):
        optimizer.zero_grad()
        index = torch.randint(0, dataset_train.length(), (1,)).item()
        loss = _linear_gyro_loss(dataset_train, index, A, b, device)
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss", loss.item(), epoch)

        if epoch % val_freq == 0:
            last_val = _validate_linear_gyro(dataset_val, A, b, device)
            writer.add_scalar("val_loss", last_val, epoch)
            print(f"Axb gyro epoch: {epoch}, train_loss: {loss.item()}, val_loss: {last_val}")

        if epoch % save_freq == 0:
            _save_ab(os.path.join(save_dir, f"model_epoch_{epoch}.pt"), A, b, epoch, last_val)

    final_epoch = max(0, epochs - 1)
    final_path = os.path.join(save_dir, f"model_epoch_{final_epoch}.pt")
    if epochs > 0 and not os.path.exists(final_path):
        _save_ab(final_path, A, b, final_epoch, last_val)
    writer.close()
    return final_path


def train_linear_acc(
    dataset_train: BaseDataset,
    dataset_val: BaseDataset,
    outputdir: str,
    gyro_path: str,
    epochs: int,
    device: str,
    lr: float,
    weight_decay: float,
    val_freq: int,
    save_freq: int,
) -> str:
    torch.manual_seed(3407)
    gyro = torch.load(gyro_path, map_location=device, weights_only=False)
    A_gyro = gyro["A"].to(device)
    b_gyro = gyro["b"].to(device)
    A = torch.nn.Parameter((torch.eye(3) + 0.01 * torch.randn(3, 3)).to(device))
    b = torch.nn.Parameter(0.01 * torch.randn(3).to(device))
    optimizer = torch.optim.Adam([A, b], lr=lr, weight_decay=weight_decay)
    writer = SummaryWriter(os.path.join(outputdir, "AccAxb_logs"))
    save_dir = os.path.join(outputdir, "AccAxbweights")
    os.makedirs(save_dir, exist_ok=True)

    last_val = float("inf")
    for epoch in range(epochs):
        optimizer.zero_grad()
        index = torch.randint(0, dataset_train.length(), (1,)).item()
        loss = _linear_acc_loss(dataset_train, index, A_gyro, b_gyro, A, b, device)
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss", loss.item(), epoch)

        if epoch % val_freq == 0:
            last_val = _validate_linear_acc(dataset_val, A_gyro, b_gyro, A, b, device)
            writer.add_scalar("val_loss", last_val, epoch)
            print(f"Axb acc epoch: {epoch}, train_loss: {loss.item()}, val_loss: {last_val}")

        if epoch % save_freq == 0:
            _save_ab(os.path.join(save_dir, f"model_epoch_{epoch}.pt"), A, b, epoch, last_val)

    final_epoch = max(0, epochs - 1)
    final_path = os.path.join(save_dir, f"model_epoch_{final_epoch}.pt")
    if epochs > 0 and not os.path.exists(final_path):
        _save_ab(final_path, A, b, final_epoch, last_val)
    writer.close()
    return final_path


def test_linear_baseline(
    dataset_test: BaseDataset,
    outputdir: str,
    gyro_path: str,
    acc_path: str,
    device: str = "cpu",
    recompute: bool = False,
) -> None:
    gyro = torch.load(gyro_path, map_location=device, weights_only=False)
    acc = torch.load(acc_path, map_location=device, weights_only=False)
    A_gyro = gyro["A"].to(device)
    b_gyro = gyro["b"].to(device)
    A_acc = acc["A"].to(device)
    b_acc = acc["b"].to(device)
    g_const = torch.tensor([0, 0, -9.81], device=device)

    with torch.no_grad():
        for index in range(dataset_test.length()):
            seq_name = dataset_test.sequences[index]
            results_path = os.path.join(outputdir, "Axb_results", seq_name, "Axb_results.p")
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            if os.path.exists(results_path) and not recompute:
                continue
            print(f"Testing Axb sequence: {seq_name}")
            _, u_imu, X_gt, t_gt = dataset_test.get_full_trajectory(index, device=device)
            w_hat = (A_gyro @ u_imu[..., :3].unsqueeze(-1)).squeeze(-1) + b_gyro
            a_hat = (A_acc @ u_imu[..., 3:].unsqueeze(-1)).squeeze(-1) + b_acc
            X_Axb = torch.zeros_like(X_gt)
            X_Axb[0] = X_gt[0]
            for i in range(1, X_gt.shape[0]):
                dt = t_gt[i] - t_gt[i - 1]
                X_Axb[i, :3, :3] = X_Axb[i - 1, :3, :3] @ Lie.SO3exp(w_hat[i - 1] * dt)
                X_Axb[i, :3, 3] = X_Axb[i - 1, :3, 3] + (
                    bmv(X_Axb[i - 1, :3, :3], a_hat[i - 1]) + g_const
                ) * dt
                X_Axb[i, :3, 4] = X_Axb[i - 1, :3, 4] + X_Axb[i - 1, :3, 3] * dt
            pdump({"X_Axb": X_Axb, "X_gt": X_gt, "t_gt": t_gt, "u_Axb": torch.cat([w_hat, a_hat], dim=-1)}, results_path)


def _linear_gyro_loss(dataset: BaseDataset, index: int, A, b, device: str) -> torch.Tensor:
    (_, X_gt_batch, _), _, _, _, starts = dataset.get_data_for_biasDy(index, device)
    _, u_imu, _, _ = dataset.get_full_trajectory(index, device)
    u_batch = torch.stack([u_imu[starts + i] for i in range(dataset._loss_window)], dim=0)
    w_hat = (A @ u_batch[..., :3].unsqueeze(-1)).squeeze(-1) + b
    R_hat = X_gt_batch[..., :3, :3].clone()
    R_t = R_hat[0].clone()
    for i in range(dataset._loss_window - 1):
        R_t = R_t @ Lie.SO3exp(w_hat[i] * dataset.dt)
        R_hat[i + 1] = R_t
    return 1e6 * SO3log_loss(R_hat, X_gt_batch[..., :3, :3])


def _linear_acc_loss(dataset: BaseDataset, index: int, A_gyro, b_gyro, A, b, device: str) -> torch.Tensor:
    (_, X_gt_batch, _), _, _, _, starts = dataset.get_data_for_biasDy(index, device)
    _, u_imu, _, _ = dataset.get_full_trajectory(index, device)
    u_batch = torch.stack([u_imu[starts + i] for i in range(dataset._loss_window)], dim=0)
    w_hat = (A_gyro @ u_batch[..., :3].unsqueeze(-1)).squeeze(-1) + b_gyro
    a_hat = (A @ u_batch[..., 3:].unsqueeze(-1)).squeeze(-1) + b
    R_hat = X_gt_batch[..., :3, :3].clone()
    v_hat = X_gt_batch[..., :3, 3].clone()
    p_hat = X_gt_batch[..., :3, 4].clone()
    R_t = R_hat[0].clone()
    v_t = v_hat[0].clone()
    p_t = p_hat[0].clone()
    g_const = torch.tensor([0, 0, -9.81], device=device)
    for i in range(dataset._loss_window - 1):
        p_t = p_t + v_t * dataset.dt
        v_t = v_t + (bmv(R_t, a_hat[i]) + g_const) * dataset.dt
        R_t = R_t @ Lie.SO3exp(w_hat[i] * dataset.dt)
        p_hat[i + 1] = p_t
        v_hat[i + 1] = v_t
        R_hat[i + 1] = R_t
    return 1e6 * torch.nn.MSELoss()(v_hat, X_gt_batch[..., :3, 3]) + 1e6 * torch.nn.MSELoss()(p_hat, X_gt_batch[..., :3, 4])


def _validate_linear_gyro(dataset: BaseDataset, A, b, device: str) -> float:
    with torch.no_grad():
        return sum(_linear_gyro_loss(dataset, i, A, b, device).item() for i in range(dataset.length()))


def _validate_linear_acc(dataset: BaseDataset, A_gyro, b_gyro, A, b, device: str) -> float:
    with torch.no_grad():
        return sum(_linear_acc_loss(dataset, i, A_gyro, b_gyro, A, b, device).item() for i in range(dataset.length()))


def _save_ab(path: str, A, b, epoch: int, val_loss: float) -> None:
    torch.save({"A": A.detach().cpu(), "b": b.detach().cpu(), "epoch": epoch, "valloss": val_loss}, path)
