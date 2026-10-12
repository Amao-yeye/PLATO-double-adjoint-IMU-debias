from __future__ import annotations

import os
import sys
import json
import math
from dataclasses import asdict, replace

import torch
from torch.utils.tensorboard import SummaryWriter

BIASDY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEBIAS_ROOT = os.path.abspath(os.path.join(BIASDY_DIR, ".."))
if BIASDY_DIR not in sys.path:
    sys.path.insert(0, BIASDY_DIR)
if DEBIAS_ROOT not in sys.path:
    sys.path.insert(0, DEBIAS_ROOT)

import Interpolation
import lie_algebra as Lie
from dataset import BaseDataset
from network_for_BiadDy import VFSE23vbiasBase, ba_func_choose, bw_func_choose
from SO3diffeq import odeint_SO3

from plato.analytic_gradient import adjoint_bias_gradient, invariant_transition_matrices, weighted_se23_mse_state_gradient
from plato.bias_adjoint import accumulate_ba_euler_adjoint_gradients
from plato.covariance import (
    build_se23_precision,
    contract_interval_covariance_state_forward_sensitivity_chunked,
    preintegrate_interval_covariance_forward_sensitivity,
    preintegrate_interval_covariance_state_vjp,
    preintegrate_supervision_covariances,
    preintegrate_supervision_covariances_forward_sensitivity,
)
from plato.gyro_adjoint import accumulate_bg_euler_adjoint_gradients
from plato.config import PlatoTrainConfig
from plato.rollout import (
    integrate_batch_paper_zoh as _integrate_batch_paper_zoh,
    evaluate_corrected_imu_and_bias_rates as _evaluate_corrected_imu_and_bias_rates,
)
from plato.losses import (
    pose_block_mse,
    pose_global_residual,
    pose_mse_loss,
    pose_right_residual,
    relative_se23_residual,
    relative_pose_residual,
    se23_block_mse,
    se23_global_residual,
    se23_gt_right_residual,
    se23_log_residual,
    se23_mse_loss,
    se23_pose_residual,
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
from plato.so3_analytic_gradient import so3_gyro_bias_gradient
from plato.supervision import (
    PoseTrajectory,
    build_gt_nll_supervised_indices,
    get_gt_se23_supervision_batch,
    get_vo_pose_supervision_batch,
    get_vo_rate_gt_se23_supervision_batch,
    get_vo_se23_supervision_batch,
)


GT_FULL_STATE_SUPERVISION = ("gt_se23", "vo_rate_gt_se23")


def _set_model_gravity_from_dataset(model, dataset: BaseDataset) -> None:
    gravity = getattr(dataset, "gravity", None)
    if gravity is None:
        gravity = getattr(dataset, "g_const", None)
    if gravity is None:
        gravity = (0.0, 0.0, -9.81)
    model.set_gravity(gravity)


def train_mse_gyro(
    dataset_train: BaseDataset,
    dataset_val: BaseDataset,
    config: PlatoTrainConfig,
    lr: float,
    weight_decay: float,
    epoch_count: int,
    gyro_checkpoint: str | None = None,
    acc_checkpoint: str | None = None,
    vo_train: dict[str, PoseTrajectory] | None = None,
    vo_val: dict[str, PoseTrajectory] | None = None,
) -> str:
    torch.manual_seed(3407)
    device = config.device
    model_save_path = os.path.join(config.outputdir, "Gyro_weights")
    os.makedirs(model_save_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(config.outputdir, "Gyro_logs"))

    if gyro_checkpoint is None:
        bw_func = bw_func_choose(config.network_type_Gyro, device)
    else:
        bw_func = load_bw_func(config.network_type_Gyro, gyro_checkpoint, device)
    if config.gyro_train_with_acc:
        if acc_checkpoint is None:
            raise ValueError("acc_checkpoint is required when gyro_train_with_acc=True")
        ba_func = load_ba_func(config.network_type_Acc, acc_checkpoint, device)
        model = VFSE23vbiasBase(biasfunc_w=bw_func, biasfunc_a=ba_func, device=device).to(device)
        model.set_ba_zero(False)
        model.freeze_func_ba()
    else:
        model = VFSE23vbiasBase(biasfunc_w=bw_func, device=device).to(device)
        model.set_ba_zero(True)
    _set_model_gravity_from_dataset(model, dataset_train)
    model.set_bw_zero(False)
    _reset_peak_memory(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    best_loss = float("inf")
    last_loss_total = float("inf")

    for epoch in range(epoch_count):
        model.train()
        optimizer.zero_grad()
        if config.gradient_mode == "autograd":
            loss, metrics, analytic_grad_norm = _train_gyro_mse_step(
                model,
                dataset_train,
                config,
                vo_train,
                step=epoch,
                writer=writer,
            )
        elif config.gradient_mode == "double_adjoint":
            loss, metrics, analytic_grad_norm = _train_gyro_mse_double_adjoint_step(
                model,
                dataset_train,
                config,
                vo_train,
                step=epoch,
                writer=writer,
            )
        else:
            raise ValueError("gradient_mode must be autograd or double_adjoint")
        optimizer.step()
        scheduler.step()

        _write_metrics(writer, "train", metrics, loss, epoch)
        if analytic_grad_norm is not None:
            writer.add_scalar("train/analytic_grad_bg_norm", analytic_grad_norm.item(), epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch % config.val_freq == 0:
            last_loss_total, val_window_losses = _validate_gyro_mse(
                model, dataset_val, config, vo_by_seq=vo_val
            )
            writer.add_scalar("val_loss", last_loss_total, epoch)
            for window_steps, window_loss in val_window_losses.items():
                writer.add_scalar(f"val_w{window_steps}/loss", window_loss, epoch)
            print(f"epoch: {epoch}, train_loss: {loss.item()}, val_loss: {last_loss_total}")
            if last_loss_total < best_loss:
                best_loss = last_loss_total
                _save_checkpoint(
                    os.path.join(model_save_path, "best_val_model.pt"),
                    config,
                    "mse_gyro",
                    epoch,
                    last_loss_total,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

        if epoch % config.save_freq == 0:
            _save_checkpoint(
                os.path.join(model_save_path, f"model_epoch_{epoch}.pt"),
                config,
                "mse_gyro",
                epoch,
                last_loss_total,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    writer.close()
    _write_peak_memory_summary(config.outputdir, "mse_gyro", device)
    final_epoch = max(0, epoch_count - 1)
    final_path = os.path.join(model_save_path, f"model_epoch_{final_epoch}.pt")
    if epoch_count > 0 and not os.path.exists(final_path):
        _save_checkpoint(
            final_path,
            config,
            "mse_gyro",
            final_epoch,
            last_loss_total,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    return final_path


def _train_gyro_mse_step(
    model,
    dataset_train: BaseDataset,
    config: PlatoTrainConfig,
    vo_train: dict[str, PoseTrajectory] | None,
    step: int,
    writer: SummaryWriter,
):
    window_steps, window_weights = _gyro_window_schedule(config)
    total_loss = None
    total_metrics: dict[str, torch.Tensor] = {}
    total_weight = 0.0

    for window, weight in zip(window_steps, window_weights):
        window_config = replace(config, window_steps=window)
        batch = _sample_batch(dataset_train, window_config, vo_train, step=step)
        solution, R_sol = _integrate_batch(model, dataset_train, batch, window_config)
        loss, metrics = _mse_loss_for_stage(batch, solution, R_sol, window_config, stage="gyro")
        scaled_loss = loss * weight
        scaled_loss.backward()

        detached_loss = scaled_loss.detach()
        total_loss = detached_loss if total_loss is None else total_loss + detached_loss
        total_weight += float(weight)
        for name, value in metrics.items():
            weighted_value = value.detach() * float(weight)
            total_metrics[name] = total_metrics.get(name, torch.zeros_like(weighted_value)) + weighted_value

        if len(window_steps) > 1:
            _write_metrics(writer, f"train_w{window}", metrics, loss.detach(), step)

    if total_loss is None:
        raise ValueError("No gyro supervision windows configured")
    averaged_metrics = {
        name: value / max(total_weight, 1e-12)
        for name, value in total_metrics.items()
    }
    return total_loss, averaged_metrics, None


def _train_gyro_mse_double_adjoint_step(
    model,
    dataset_train: BaseDataset,
    config: PlatoTrainConfig,
    vo_train: dict[str, PoseTrajectory] | None,
    step: int,
    writer: SummaryWriter,
):
    window_steps, window_weights = _gyro_window_schedule(config)
    total_loss = None
    total_metrics: dict[str, torch.Tensor] = {}
    total_weight = 0.0
    grad_norm_sq = torch.zeros((), device=config.device)

    for window, weight in zip(window_steps, window_weights):
        window_config = replace(config, window_steps=window)
        batch = _sample_batch(dataset_train, window_config, vo_train, step=step)
        loss, metrics, grad_norm = _train_gyro_mse_double_adjoint_window(
            model,
            dataset_train,
            batch,
            window_config,
            weight=float(weight),
        )

        detached_loss = loss.detach()
        total_loss = detached_loss if total_loss is None else total_loss + detached_loss
        total_weight += float(weight)
        for name, value in metrics.items():
            weighted_value = value.detach() * float(weight)
            total_metrics[name] = total_metrics.get(name, torch.zeros_like(weighted_value)) + weighted_value
        grad_norm_sq = grad_norm_sq + grad_norm.square()

        if len(window_steps) > 1:
            _write_metrics(writer, f"train_w{window}", metrics, loss.detach(), step)
            writer.add_scalar(f"train_w{window}/analytic_grad_bg_norm", grad_norm.item(), step)

    if total_loss is None:
        raise ValueError("No gyro supervision windows configured")
    averaged_metrics = {
        name: value / max(total_weight, 1e-12)
        for name, value in total_metrics.items()
    }
    return total_loss, averaged_metrics, grad_norm_sq.sqrt().detach()


def _train_gyro_mse_double_adjoint_window(
    model,
    dataset_train: BaseDataset,
    batch,
    config: PlatoTrainConfig,
    weight: float = 1.0,
):
    if config.supervision not in GT_FULL_STATE_SUPERVISION:
        raise NotImplementedError(
            "gyro double_adjoint supports only GT full-state supervision"
        )
    if config.integral_method != "euler":
        raise NotImplementedError("gyro double_adjoint currently supports only integral_method='euler'")
    if config.mse_residual_mode != "right":
        raise NotImplementedError("gyro double_adjoint requires mse_residual_mode='right'")

    sup = batch.supervised_indices

    with torch.no_grad():
        solution, R_sol = _integrate_batch(model, dataset_train, batch, config)
        loss_unweighted, metrics = _mse_loss_for_stage(batch, solution, R_sol, config, stage="gyro")

    R_ref = _se23_ref_at_supervision(batch)[..., :3, :3]
    gyro_gradient_kwargs = {}
    if getattr(config, "mean_discretization", "legacy_euler") == "paper_zoh":
        corrected_omega, _ = _corrected_imu_sequence(model, solution, R_sol)
        gyro_gradient_kwargs = {
            "discretization": "paper_zoh",
            "corrected_omega": corrected_omega,
        }
    analytic = so3_gyro_bias_gradient(
        R_sol.detach(),
        R_ref.detach(),
        t=batch.t_odeint.detach(),
        rot_weight=config.mse_rot_weight,
        residual_jacobian=config.analytic_residual_jacobian,
        invariant_error="right",
        supervised_indices=sup,
        **gyro_gradient_kwargs,
    )
    grad_bg = (
        1e6
        * float(weight)
        * float(config.analytic_grad_scale)
        * _gyro_objective_gradient_scale(config)
        * analytic.bias_gradient
    )
    if config.analytic_grad_clip_norm > 0:
        grad_norm = grad_bg.norm()
        if grad_norm > config.analytic_grad_clip_norm:
            grad_bg = grad_bg * (config.analytic_grad_clip_norm / (grad_norm + 1e-12))

    accumulate_bg_euler_adjoint_gradients(
        model,
        solution,
        batch.t_odeint,
        grad_bg,
        jump_indices=torch.arange(grad_bg.shape[0], device=grad_bg.device),
    )
    return (float(weight) * loss_unweighted).detach(), metrics, grad_bg.norm().detach()


def train_mse_acc(
    dataset_train: BaseDataset,
    dataset_val: BaseDataset,
    config: PlatoTrainConfig,
    gyro_checkpoint: str,
    lr: float,
    weight_decay: float,
    epoch_count: int,
    acc_checkpoint: str | None = None,
    vo_train: dict[str, PoseTrajectory] | None = None,
    vo_val: dict[str, PoseTrajectory] | None = None,
) -> str:
    torch.manual_seed(3407)
    device = config.device
    model_save_path = os.path.join(config.outputdir, "Acc_weights")
    os.makedirs(model_save_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(config.outputdir, "Acc_logs"))

    bw_func = load_bw_func(config.network_type_Gyro, gyro_checkpoint, device)
    if acc_checkpoint is None:
        ba_func = ba_func_choose(config.network_type_Acc, device)
    else:
        ba_func = load_ba_func(config.network_type_Acc, acc_checkpoint, device)
    model = VFSE23vbiasBase(biasfunc_w=bw_func, biasfunc_a=ba_func, device=device).to(device)
    _set_model_gravity_from_dataset(model, dataset_train)
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    model.freeze_func_bw()
    _reset_peak_memory(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    best_loss = float("inf")
    last_loss_total = float("inf")

    for epoch in range(epoch_count):
        model.train()
        optimizer.zero_grad()
        batch = _sample_batch(dataset_train, config, vo_train, step=epoch)
        if config.gradient_mode == "autograd":
            solution, R_sol = _integrate_batch(model, dataset_train, batch, config)
            loss, metrics = _mse_loss_for_stage(batch, solution, R_sol, config, stage="acc")
            loss.backward()
            analytic_grad_norm = None
        elif config.gradient_mode == "double_adjoint":
            loss, metrics, analytic_grad_norm = _train_acc_mse_double_adjoint_step(
                model,
                dataset_train,
                batch,
                config,
            )
        else:
            raise ValueError("gradient_mode must be autograd or double_adjoint")
        optimizer.step()
        ba_rate_limit = _apply_ba_rate_continuation(
            model.biasfunc_a,
            step=epoch,
            total_steps=epoch_count,
            config=config,
        )
        scheduler.step()

        _write_metrics(writer, "train", metrics, loss, epoch)
        if analytic_grad_norm is not None:
            writer.add_scalar("train/analytic_grad_ba_norm", analytic_grad_norm.item(), epoch)
        if ba_rate_limit is not None:
            writer.add_scalar("train/ba_rate_limit", ba_rate_limit, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch % config.val_freq == 0:
            last_loss_total = _validate_mse(model, dataset_val, config, stage="acc", vo_by_seq=vo_val)
            writer.add_scalar("val_loss", last_loss_total, epoch)
            print(f"epoch: {epoch}, train_loss: {loss.item()}, val_loss: {last_loss_total}")
            if last_loss_total < best_loss:
                best_loss = last_loss_total
                _save_checkpoint(
                    os.path.join(model_save_path, "best_val_model.pt"),
                    config,
                    "mse_acc",
                    epoch,
                    last_loss_total,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

        if epoch % config.save_freq == 0:
            _save_checkpoint(
                os.path.join(model_save_path, f"model_epoch_{epoch}.pt"),
                config,
                "mse_acc",
                epoch,
                last_loss_total,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    writer.close()
    _write_peak_memory_summary(config.outputdir, "mse_acc", device)
    final_epoch = max(0, epoch_count - 1)
    final_path = os.path.join(model_save_path, f"model_epoch_{final_epoch}.pt")
    if epoch_count > 0 and not os.path.exists(final_path):
        _save_checkpoint(
            final_path,
            config,
            "mse_acc",
            final_epoch,
            last_loss_total,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    return final_path


def _train_acc_mse_double_adjoint_step(
    model,
    dataset_train: BaseDataset,
    batch,
    config: PlatoTrainConfig,
):
    if config.supervision not in GT_FULL_STATE_SUPERVISION:
        raise NotImplementedError(
            "acc double_adjoint supports only GT full-state supervision"
        )
    if config.integral_method != "euler":
        raise NotImplementedError("double_adjoint currently supports only integral_method='euler'")
    if config.mse_residual_mode != "right":
        raise NotImplementedError("double_adjoint requires mse_residual_mode='right'")

    sup = batch.supervised_indices

    with torch.no_grad():
        solution, R_sol = _integrate_batch(model, dataset_train, batch, config)

    X_ref = _se23_ref_at_supervision(batch)
    residual = se23_gt_right_residual(
        X_ref,
        R_sol[sup],
        solution[sup],
    )
    blocks = _mse_blocks_for_stage("acc", full_state=True, config=config)
    rot_weight, vel_weight, pos_weight = _mse_weights_for_stage("acc", config)
    loss = 1e6 * se23_mse_loss(
        residual,
        rot_weight=rot_weight,
        vel_weight=vel_weight,
        pos_weight=pos_weight,
        blocks=blocks,
    )
    metrics = se23_block_mse(residual)

    supervised_state_grad = weighted_se23_mse_state_gradient(
        residual.detach(),
        rot_weight=rot_weight,
        vel_weight=vel_weight,
        pos_weight=pos_weight,
        blocks=blocks,
        residual_jacobian=config.analytic_residual_jacobian,
    )
    state_grad = torch.zeros(
        solution.shape[0],
        *solution.shape[1:-1],
        9,
        device=solution.device,
        dtype=solution.dtype,
    )
    state_grad.index_add_(0, sup, supervised_state_grad)
    F, G = _invariant_transition_matrices_for_rollout(
        model,
        solution,
        R_sol,
        batch.t_odeint,
        config,
    )
    analytic = adjoint_bias_gradient(state_grad, F, G, detach=True)
    # Both the first-order G and finite-step B use columns [ba, bg].
    grad_ba = 1e6 * float(config.analytic_grad_scale) * analytic.bias_gradient[..., 0:3]
    if config.analytic_grad_clip_norm > 0:
        grad_norm = grad_ba.norm()
        if grad_norm > config.analytic_grad_clip_norm:
            grad_ba = grad_ba * (config.analytic_grad_clip_norm / (grad_norm + 1e-12))

    accumulate_ba_euler_adjoint_gradients(
        model,
        solution,
        batch.t_odeint,
        grad_ba,
        jump_indices=torch.arange(grad_ba.shape[0], device=grad_ba.device),
    )
    return loss.detach(), metrics, grad_ba.norm().detach()


def train_mse_joint(
    dataset_train: BaseDataset,
    dataset_val: BaseDataset,
    config: PlatoTrainConfig,
    gyro_checkpoint: str,
    acc_checkpoint: str | None,
    lr: float,
    weight_decay: float,
    epoch_count: int,
    vo_train: dict[str, PoseTrajectory] | None = None,
    vo_val: dict[str, PoseTrajectory] | None = None,
) -> str:
    torch.manual_seed(3407)
    device = config.device
    model_save_path = os.path.join(config.outputdir, "Joint_weights")
    os.makedirs(model_save_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(config.outputdir, "Joint_logs"))

    bw_func = load_bw_func(config.network_type_Gyro, gyro_checkpoint, device)
    if acc_checkpoint is None:
        # This supports a rotation-pretrained BG warm start followed by the
        # paper's joint SE_2(3) bias-dynamics objective.  BA uses the same
        # deterministic seed/order as the ordinary staged trainer.
        ba_func = ba_func_choose(config.network_type_Acc, device)
    else:
        ba_func = load_ba_func(config.network_type_Acc, acc_checkpoint, device)
    model = VFSE23vbiasBase(biasfunc_w=bw_func, biasfunc_a=ba_func, device=device).to(device)
    _set_model_gravity_from_dataset(model, dataset_train)
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    model.unfreeze_func_bw()
    model.unfreeze_func_ba()
    _reset_peak_memory(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    best_loss = float("inf")
    last_loss_total = float("inf")

    for epoch in range(epoch_count):
        model.train()
        optimizer.zero_grad()
        batch = _sample_batch(dataset_train, config, vo_train, step=epoch)
        if config.gradient_mode == "autograd":
            solution, R_sol = _integrate_batch(model, dataset_train, batch, config)
            loss, metrics = _mse_loss_for_stage(
                batch, solution, R_sol, config, stage="joint"
            )
            loss.backward()
            analytic_grad_norms = {}
        elif config.gradient_mode == "double_adjoint":
            loss, metrics, analytic_grad_norms = _train_joint_mse_double_adjoint_step(
                model,
                dataset_train,
                batch,
                config,
            )
        else:
            raise ValueError("gradient_mode must be autograd or double_adjoint")
        optimizer.step()
        ba_rate_limit = _project_ba_rate_limit(
            model.biasfunc_a, config.ba_rate_final_limit
        )
        scheduler.step()

        _write_metrics(writer, "train", metrics, loss, epoch)
        for name, value in analytic_grad_norms.items():
            writer.add_scalar(f"train/{name}", value.item(), epoch)
        if ba_rate_limit is not None:
            writer.add_scalar("train/ba_rate_limit", ba_rate_limit, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch % config.val_freq == 0:
            last_loss_total = _validate_mse(
                model, dataset_val, config, stage="joint", vo_by_seq=vo_val
            )
            writer.add_scalar("val_loss", last_loss_total, epoch)
            print(f"epoch: {epoch}, train_loss: {loss.item()}, val_loss: {last_loss_total}")
            if last_loss_total < best_loss:
                best_loss = last_loss_total
                _save_checkpoint(
                    os.path.join(model_save_path, "best_val_model.pt"),
                    config,
                    "mse_joint",
                    epoch,
                    last_loss_total,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

        if epoch % config.save_freq == 0:
            _save_checkpoint(
                os.path.join(model_save_path, f"model_epoch_{epoch}.pt"),
                config,
                "mse_joint",
                epoch,
                last_loss_total,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    writer.close()
    _write_peak_memory_summary(config.outputdir, "mse_joint", device)
    final_epoch = max(0, epoch_count - 1)
    final_path = os.path.join(model_save_path, f"model_epoch_{final_epoch}.pt")
    if epoch_count > 0 and not os.path.exists(final_path):
        _save_checkpoint(
            final_path,
            config,
            "mse_joint",
            final_epoch,
            last_loss_total,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    return final_path


def _train_joint_mse_double_adjoint_step(
    model,
    dataset_train: BaseDataset,
    batch,
    config: PlatoTrainConfig,
):
    """Differentiate joint SE_2(3) MSE without retaining the ODE rollout graph.

    The local loss derivative is taken with respect to a left-invariant state
    perturbation.  The outer discrete invariant-error adjoint maps this state
    gradient to the BA/BG trajectories, and the two inner Euler adjoints map
    those trajectory gradients to the corresponding network parameters.
    """
    if config.integral_method != "euler":
        raise NotImplementedError(
            "joint MSE double_adjoint currently supports only integral_method='euler'"
        )

    with torch.no_grad():
        solution, R_sol = _integrate_batch(model, dataset_train, batch, config)

    delta = torch.zeros(
        *solution.shape[:-1],
        9,
        device=solution.device,
        dtype=solution.dtype,
        requires_grad=True,
    )
    R_perturbed, solution_perturbed = _apply_se23_left_perturbation(
        R_sol.detach(),
        solution.detach(),
        delta,
    )
    loss, metrics = _mse_loss_for_stage(
        batch,
        solution_perturbed,
        R_perturbed,
        config,
        stage="joint",
    )
    state_grad = torch.autograd.grad(
        loss,
        delta,
        retain_graph=False,
        create_graph=False,
    )[0].detach()

    train_ba = _has_trainable_parameters(model.biasfunc_a)
    train_bg = _has_trainable_parameters(model.biasfunc_w)
    grad_norms = _accumulate_se23_bias_adjoint_gradients(
        model,
        solution,
        R_sol,
        batch.t_odeint,
        state_grad,
        config,
        train_ba=train_ba,
        train_bg=train_bg,
    )
    return loss.detach(), _detach_metrics(metrics), grad_norms


def train_nll(
    dataset_train: BaseDataset,
    dataset_val: BaseDataset,
    config: PlatoTrainConfig,
    gyro_checkpoint: str,
    acc_checkpoint: str,
    lr: float,
    weight_decay: float,
    epoch_count: int,
    nll_freeze_gyro: bool = False,
    nll_freeze_acc: bool = False,
    noise_checkpoint: str | None = None,
    nll_freeze_covariance: bool = False,
    nll_covariance_only: bool = False,
    vo_train: dict[str, PoseTrajectory] | None = None,
    vo_val: dict[str, PoseTrajectory] | None = None,
) -> str:
    torch.manual_seed(3407)
    device = config.device
    model_save_path = os.path.join(config.outputdir, "NLL_weights")
    os.makedirs(model_save_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(config.outputdir, "NLL_logs"))

    bw_func = load_bw_func(config.network_type_Gyro, gyro_checkpoint, device)
    ba_func = load_ba_func(config.network_type_Acc, acc_checkpoint, device)
    model = VFSE23vbiasBase(biasfunc_w=bw_func, biasfunc_a=ba_func, device=device).to(device)
    _set_model_gravity_from_dataset(model, dataset_train)
    model.set_bw_zero(False)
    model.set_ba_zero(False)
    if nll_covariance_only:
        for param in model.parameters():
            param.requires_grad = False
    elif nll_freeze_gyro:
        model.freeze_func_bw()
    else:
        model.unfreeze_func_bw()
    if not nll_covariance_only:
        if nll_freeze_acc:
            model.freeze_func_ba()
        else:
            model.unfreeze_func_ba()

    noise_params = PlatoNoiseParams(
        init_sigma_a=config.init_sigma_a,
        init_sigma_g=config.init_sigma_g,
        isotropic=config.plato_noise_isotropic,
        dtype=torch.float64,
        device=device,
    )
    if noise_checkpoint is not None:
        checkpoint = torch.load(noise_checkpoint, map_location=device, weights_only=False)
        if "noise_params" not in checkpoint:
            raise KeyError(f"Checkpoint has no noise_params: {noise_checkpoint}")
        noise_params.load_state_dict(checkpoint["noise_params"])
    noise_params.clamp_sigmas(
        min_sigma_a=config.nll_min_sigma_a,
        max_sigma_a=config.nll_max_sigma_a,
        min_sigma_g=config.nll_min_sigma_g,
        max_sigma_g=config.nll_max_sigma_g,
    )
    if nll_freeze_covariance:
        for param in noise_params.parameters():
            param.requires_grad = False
    if nll_covariance_only and nll_freeze_covariance:
        raise ValueError("nll_covariance_only cannot be combined with nll_freeze_covariance")
    trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
    trainable += list(filter(lambda p: p.requires_grad, noise_params.parameters()))
    if not trainable:
        raise ValueError("No trainable parameters selected for NLL training")
    _reset_peak_memory(device)
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")
    last_loss_total = float("inf")

    for epoch in range(epoch_count):
        model.train()
        optimizer.zero_grad()
        batch = _sample_batch(dataset_train, config, vo_train, step=epoch)
        if config.gradient_mode == "autograd":
            solution, R_sol = _integrate_batch(model, dataset_train, batch, config)
            loss, metrics = _nll_loss_for_batch(batch, solution, R_sol, config, noise_params)
            loss.backward()
            analytic_grad_norms = {}
        elif config.gradient_mode == "double_adjoint":
            loss, metrics, analytic_grad_norms = _train_nll_double_adjoint_step(
                model,
                dataset_train,
                batch,
                config,
                noise_params,
            )
        else:
            raise ValueError("gradient_mode must be autograd or double_adjoint")
        optimizer.step()
        _project_ba_rate_limit(
            model.biasfunc_a, config.ba_rate_final_limit
        )
        noise_params.clamp_sigmas(
            min_sigma_a=config.nll_min_sigma_a,
            max_sigma_a=config.nll_max_sigma_a,
            min_sigma_g=config.nll_min_sigma_g,
            max_sigma_g=config.nll_max_sigma_g,
        )

        _write_metrics(writer, "train", metrics, loss, epoch)
        for name, value in analytic_grad_norms.items():
            writer.add_scalar(f"train/{name}", value.item(), epoch)
        writer.add_scalar("train/log_sigma_a", noise_params.log_sigma_a.mean().item(), epoch)
        writer.add_scalar("train/log_sigma_g", noise_params.log_sigma_g.mean().item(), epoch)
        sigma_a, sigma_g = noise_params.sigmas()
        writer.add_scalar("train/sigma_a", sigma_a.mean().item(), epoch)
        writer.add_scalar("train/sigma_g", sigma_g.mean().item(), epoch)

        if epoch % config.val_freq == 0:
            last_loss_total, val_metrics = _validate_nll(
                model, noise_params, dataset_val, config, vo_by_seq=vo_val
            )
            writer.add_scalar("val_loss", last_loss_total, epoch)
            _write_scalar_metrics(writer, "val", val_metrics, epoch)
            print(f"epoch: {epoch}, train_nll: {loss.item()}, val_loss: {last_loss_total}")
            if last_loss_total < best_loss:
                best_loss = last_loss_total
                _save_checkpoint(
                    os.path.join(model_save_path, "best_val_model.pt"),
                    config,
                    "nll",
                    epoch,
                    last_loss_total,
                    model=model,
                    noise_params=noise_params,
                    optimizer=optimizer,
                )

        if epoch % config.save_freq == 0:
            _save_checkpoint(
                os.path.join(model_save_path, f"model_epoch_{epoch}.pt"),
                config,
                "nll",
                epoch,
                last_loss_total,
                model=model,
                noise_params=noise_params,
                optimizer=optimizer,
            )

    writer.close()
    _write_peak_memory_summary(config.outputdir, "nll", device)
    final_epoch = max(0, epoch_count - 1)
    final_path = os.path.join(model_save_path, f"model_epoch_{final_epoch}.pt")
    if epoch_count > 0 and not os.path.exists(final_path):
        _save_checkpoint(
            final_path,
            config,
            "nll",
            final_epoch,
            last_loss_total,
            model=model,
            noise_params=noise_params,
            optimizer=optimizer,
        )
    return final_path


def train_nll_alternating(
    dataset_train: BaseDataset,
    dataset_val: BaseDataset,
    config: PlatoTrainConfig,
    gyro_checkpoint: str,
    acc_checkpoint: str,
    bias_lr: float,
    covariance_lr: float,
    bias_weight_decay: float,
    covariance_weight_decay: float,
    epoch_count: int,
    block_epochs: int = 1,
    start_with: str = "covariance",
    noise_checkpoint: str | None = None,
    vo_train: dict[str, PoseTrajectory] | None = None,
    vo_val: dict[str, PoseTrajectory] | None = None,
) -> str:
    """Alternate joint BA/BG and covariance NLL optimizer steps.

    A bias phase jointly updates both bias NODEs with the double adjoint while
    holding ``psi`` fixed.  A covariance phase holds the complete nominal
    bias rollout fixed and updates only ``psi`` using forward sensitivity.
    ``epoch_count`` counts total optimizer steps across both parameter blocks.
    """
    if config.gradient_mode != "double_adjoint":
        raise ValueError("alternating NLL bias phases require gradient_mode='double_adjoint'")
    if config.covariance_gradient_mode != "forward_sensitivity":
        raise ValueError(
            "alternating NLL covariance phases require "
            "covariance_gradient_mode='forward_sensitivity'"
        )
    if epoch_count < 1:
        raise ValueError("epoch_count must be positive")
    if block_epochs < 1:
        raise ValueError("block_epochs must be positive")
    if start_with not in ("covariance", "bias"):
        raise ValueError("start_with must be covariance or bias")

    torch.manual_seed(3407)
    device = config.device
    model_save_path = os.path.join(config.outputdir, "NLL_alternating_weights")
    os.makedirs(model_save_path, exist_ok=True)
    writer = SummaryWriter(os.path.join(config.outputdir, "NLL_alternating_logs"))

    bw_func = load_bw_func(config.network_type_Gyro, gyro_checkpoint, device)
    ba_func = load_ba_func(config.network_type_Acc, acc_checkpoint, device)
    model = VFSE23vbiasBase(
        biasfunc_w=bw_func, biasfunc_a=ba_func, device=device
    ).to(device)
    _set_model_gravity_from_dataset(model, dataset_train)
    model.set_bw_zero(False)
    model.set_ba_zero(False)

    noise_params = PlatoNoiseParams(
        init_sigma_a=config.init_sigma_a,
        init_sigma_g=config.init_sigma_g,
        isotropic=config.plato_noise_isotropic,
        dtype=torch.float64,
        device=device,
    )
    if noise_checkpoint is not None:
        checkpoint = torch.load(noise_checkpoint, map_location=device, weights_only=False)
        if "noise_params" not in checkpoint:
            raise KeyError(f"Checkpoint has no noise_params: {noise_checkpoint}")
        noise_params.load_state_dict(checkpoint["noise_params"])
    noise_params.clamp_sigmas(
        min_sigma_a=config.nll_min_sigma_a,
        max_sigma_a=config.nll_max_sigma_a,
        min_sigma_g=config.nll_min_sigma_g,
        max_sigma_g=config.nll_max_sigma_g,
    )

    bias_parameters = [
        *model.biasfunc_w.parameters(),
        *model.biasfunc_a.parameters(),
    ]
    covariance_parameters = list(noise_params.parameters())
    bias_optimizer = torch.optim.Adam(
        bias_parameters, lr=bias_lr, weight_decay=bias_weight_decay
    )
    covariance_optimizer = torch.optim.Adam(
        covariance_parameters,
        lr=covariance_lr,
        weight_decay=covariance_weight_decay,
    )
    _reset_peak_memory(device)

    best_loss = float("inf")
    last_loss_total = float("inf")
    phase_counts = {"bias": 0, "covariance": 0}

    for epoch in range(epoch_count):
        block_index = epoch // block_epochs
        first_phase = start_with
        second_phase = "bias" if start_with == "covariance" else "covariance"
        phase = first_phase if block_index % 2 == 0 else second_phase
        phase_counts[phase] += 1

        bias_optimizer.zero_grad(set_to_none=True)
        covariance_optimizer.zero_grad(set_to_none=True)
        batch = _sample_batch(dataset_train, config, vo_train, step=epoch)
        model.train()

        if phase == "covariance":
            _set_parameters_trainable(bias_parameters, False)
            _set_parameters_trainable(covariance_parameters, True)
            with torch.no_grad():
                solution, R_sol = _integrate_batch(
                    model, dataset_train, batch, config
                )
            loss, metrics = _nll_loss_for_batch(
                batch,
                solution.detach(),
                R_sol.detach(),
                config,
                noise_params,
            )
            loss.backward()
            covariance_grad_norm = _parameter_gradient_norm(covariance_parameters)
            covariance_optimizer.step()
            noise_params.clamp_sigmas(
                min_sigma_a=config.nll_min_sigma_a,
                max_sigma_a=config.nll_max_sigma_a,
                min_sigma_g=config.nll_min_sigma_g,
                max_sigma_g=config.nll_max_sigma_g,
            )
            grad_norms = {"covariance_grad_norm": covariance_grad_norm}
        else:
            _set_parameters_trainable(bias_parameters, True)
            _set_parameters_trainable(covariance_parameters, False)
            loss, metrics, grad_norms = _train_nll_double_adjoint_step(
                model,
                dataset_train,
                batch,
                config,
                noise_params,
            )
            bias_optimizer.step()

        ba_rate_limit = _project_ba_rate_limit(
            model.biasfunc_a, config.ba_rate_final_limit
        )

        prefix = f"train_{phase}"
        _write_metrics(writer, prefix, metrics, loss, epoch)
        for name, value in grad_norms.items():
            writer.add_scalar(f"{prefix}/{name}", _scalar_value(value), epoch)
        writer.add_scalar("train/phase_is_bias", float(phase == "bias"), epoch)
        writer.add_scalar("train/bias_lr", bias_optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar(
            "train/covariance_lr", covariance_optimizer.param_groups[0]["lr"], epoch
        )
        sigma_a, sigma_g = noise_params.sigmas()
        writer.add_scalar("train/sigma_a", sigma_a.mean().item(), epoch)
        writer.add_scalar("train/sigma_g", sigma_g.mean().item(), epoch)
        if ba_rate_limit is not None:
            writer.add_scalar("train/ba_rate_limit", ba_rate_limit, epoch)

        if epoch % config.val_freq == 0:
            last_loss_total, val_metrics = _validate_nll(
                model, noise_params, dataset_val, config, vo_by_seq=vo_val
            )
            writer.add_scalar("val_loss", last_loss_total, epoch)
            _write_scalar_metrics(writer, "val", val_metrics, epoch)
            print(
                f"epoch: {epoch}, phase: {phase}, train_nll: {loss.item()}, "
                f"val_loss: {last_loss_total}, sigma_a: {sigma_a.mean().item()}, "
                f"sigma_g: {sigma_g.mean().item()}"
            )
            if last_loss_total < best_loss:
                best_loss = last_loss_total
                _save_checkpoint(
                    os.path.join(model_save_path, "best_val_model.pt"),
                    config,
                    "nll_alternating",
                    epoch,
                    last_loss_total,
                    model=model,
                    noise_params=noise_params,
                    optimizer=bias_optimizer,
                    extra_state=_alternating_checkpoint_state(
                        bias_optimizer,
                        covariance_optimizer,
                        phase,
                        phase_counts,
                        block_epochs,
                        start_with,
                    ),
                )

        if epoch % config.save_freq == 0:
            _save_checkpoint(
                os.path.join(model_save_path, f"model_epoch_{epoch}.pt"),
                config,
                "nll_alternating",
                epoch,
                last_loss_total,
                model=model,
                noise_params=noise_params,
                optimizer=bias_optimizer,
                extra_state=_alternating_checkpoint_state(
                    bias_optimizer,
                    covariance_optimizer,
                    phase,
                    phase_counts,
                    block_epochs,
                    start_with,
                ),
            )

    writer.close()
    _write_peak_memory_summary(config.outputdir, "nll_alternating", device)
    final_epoch = epoch_count - 1
    final_path = os.path.join(
        model_save_path, f"model_epoch_{final_epoch}.pt"
    )
    if not os.path.exists(final_path):
        _save_checkpoint(
            final_path,
            config,
            "nll_alternating",
            final_epoch,
            last_loss_total,
            model=model,
            noise_params=noise_params,
            optimizer=bias_optimizer,
            extra_state=_alternating_checkpoint_state(
                bias_optimizer,
                covariance_optimizer,
                phase,
                phase_counts,
                block_epochs,
                start_with,
            ),
        )
    return final_path


def _set_parameters_trainable(
    parameters: list[torch.nn.Parameter], trainable: bool
) -> None:
    for parameter in parameters:
        parameter.requires_grad_(trainable)


def _parameter_gradient_norm(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    gradients = [
        parameter.grad.detach().reshape(-1)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        device = parameters[0].device if parameters else "cpu"
        return torch.zeros((), device=device)
    return torch.cat(gradients).norm()


def _project_ba_rate_limit(ba_model, maximum: float) -> float | None:
    if not hasattr(ba_model, "project_rate_limit"):
        return None
    ba_model.project_rate_limit(float(maximum))
    return float(ba_model.rate_limit().detach().cpu())


def _apply_ba_rate_continuation(
    ba_model,
    *,
    step: int,
    total_steps: int,
    config: PlatoTrainConfig,
) -> float | None:
    """Projected homotopy for a learnable BA rate without changing the loss."""
    if not hasattr(ba_model, "project_rate_limit"):
        return None
    last_step = max(int(total_steps) - 1, 1)
    anneal_start = int(
        round(float(config.ba_rate_anneal_start_fraction) * last_step)
    )
    initial_limit = float(getattr(ba_model, "initial_rate", 0.1))
    final_limit = float(config.ba_rate_final_limit)
    if step <= anneal_start:
        maximum = initial_limit
    else:
        progress = (step - anneal_start) / max(last_step - anneal_start, 1)
        maximum = math.exp(
            (1.0 - progress) * math.log(initial_limit)
            + progress * math.log(final_limit)
        )
    return _project_ba_rate_limit(ba_model, maximum)


def _alternating_checkpoint_state(
    bias_optimizer,
    covariance_optimizer,
    phase: str,
    phase_counts: dict[str, int],
    block_epochs: int,
    start_with: str,
) -> dict:
    return {
        "optimizer_bias": bias_optimizer.state_dict(),
        "optimizer_covariance": covariance_optimizer.state_dict(),
        "alternating_protocol": {
            "last_phase": phase,
            "phase_counts": dict(phase_counts),
            "block_epochs": int(block_epochs),
            "start_with": start_with,
        },
    }


def _train_nll_double_adjoint_step(
    model,
    dataset_train: BaseDataset,
    batch,
    config: PlatoTrainConfig,
    noise_params: PlatoNoiseParams,
):
    if config.supervision not in (*GT_FULL_STATE_SUPERVISION, "vo_pose", "vo_se23"):
        raise NotImplementedError(
            "Unsupported supervision for NLL double_adjoint"
        )
    if config.integral_method != "euler":
        raise NotImplementedError("NLL double_adjoint currently supports only integral_method='euler'")

    with torch.no_grad():
        solution, R_sol = _integrate_batch(model, dataset_train, batch, config)

    delta = torch.zeros(
        *solution.shape[:-1],
        9,
        device=solution.device,
        dtype=solution.dtype,
        requires_grad=True,
    )
    R_perturbed, solution_perturbed = _apply_se23_left_perturbation(
        R_sol.detach(),
        solution.detach(),
        delta,
    )
    use_streaming_covariance = (
        config.supervision in GT_FULL_STATE_SUPERVISION
        and getattr(config, "nll_stream_covariance_intervals", True)
        and config.nll_gt_mode == "sparse"
        and config.covariance_gradient_mode == "forward_sensitivity"
    )
    if use_streaming_covariance:
        loss, metrics, state_grad = (
            _sparse_gt_nll_streaming_covariance_double_adjoint_backward(
                batch,
                solution_perturbed,
                R_perturbed,
                solution.detach(),
                R_sol.detach(),
                delta,
                config,
                noise_params,
            )
        )
    else:
        loss, metrics = _nll_loss_for_batch(
            batch,
            solution_perturbed,
            R_perturbed,
            config,
            noise_params,
            covariance_nominal=(solution.detach(), R_sol.detach()),
            covariance_state_perturbation=delta,
        )
        loss.backward()
        if delta.grad is None:
            raise RuntimeError("NLL double_adjoint failed to produce a state gradient")
        state_grad = delta.grad.detach()
    train_ba = _has_trainable_parameters(model.biasfunc_a)
    train_bg = _has_trainable_parameters(model.biasfunc_w)
    grad_norms = _accumulate_se23_bias_adjoint_gradients(
        model,
        solution,
        R_sol,
        batch.t_odeint,
        state_grad,
        config,
        train_ba=train_ba,
        train_bg=train_bg,
    )
    metrics = _detach_metrics(metrics)
    return loss.detach(), metrics, grad_norms


def _apply_se23_left_perturbation(
    R_pred: torch.Tensor,
    solution: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    X_pred = Lie.SEn3fromSO3Rn(
        R_pred,
        torch.cat([solution[..., 7:10], solution[..., 10:13]], dim=-1),
    )
    X_perturbed = torch.matmul(Lie.SEn3exp(delta), X_pred)
    solution_perturbed = solution.clone()
    solution_perturbed[..., 7:10] = X_perturbed[..., :3, 3]
    solution_perturbed[..., 10:13] = X_perturbed[..., :3, 4]
    return X_perturbed[..., :3, :3], solution_perturbed


def _accumulate_se23_bias_adjoint_gradients(
    model,
    solution: torch.Tensor,
    R_sol: torch.Tensor,
    t_odeint: torch.Tensor,
    state_grad: torch.Tensor,
    config: PlatoTrainConfig,
    train_ba: bool,
    train_bg: bool,
) -> dict[str, torch.Tensor]:
    F, G = _invariant_transition_matrices_for_rollout(
        model,
        solution,
        R_sol,
        t_odeint,
        config,
    )
    analytic = adjoint_bias_gradient(state_grad, F, G, detach=True)
    grad_norms: dict[str, torch.Tensor] = {}

    if train_ba:
        grad_ba = float(config.analytic_grad_scale) * analytic.bias_gradient[..., 0:3]
        grad_ba = _clip_adjoint_jump_gradient(grad_ba, config)
        accumulate_ba_euler_adjoint_gradients(
            model,
            solution,
            t_odeint,
            grad_ba,
            jump_indices=torch.arange(grad_ba.shape[0], device=grad_ba.device),
        )
        grad_norms["analytic_grad_ba_norm"] = grad_ba.norm().detach()

    if train_bg:
        grad_bg = float(config.analytic_grad_scale) * analytic.bias_gradient[..., 3:6]
        grad_bg = _clip_adjoint_jump_gradient(grad_bg, config)
        accumulate_bg_euler_adjoint_gradients(
            model,
            solution,
            t_odeint,
            grad_bg,
            jump_indices=torch.arange(grad_bg.shape[0], device=grad_bg.device),
        )
        grad_norms["analytic_grad_bg_norm"] = grad_bg.norm().detach()

    return grad_norms


def _clip_adjoint_jump_gradient(gradient: torch.Tensor, config: PlatoTrainConfig) -> torch.Tensor:
    if config.analytic_grad_clip_norm <= 0:
        return gradient
    grad_norm = gradient.norm()
    if grad_norm > config.analytic_grad_clip_norm:
        return gradient * (config.analytic_grad_clip_norm / (grad_norm + 1e-12))
    return gradient


def _has_trainable_parameters(module) -> bool:
    return module is not None and any(parameter.requires_grad for parameter in module.parameters())


def _detach_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach() for name, value in metrics.items()}


def _add_nees_metrics(metrics: dict[str, torch.Tensor], nees: torch.Tensor, dof: int) -> None:
    dof_tensor = torch.as_tensor(float(dof), device=nees.device, dtype=nees.dtype)
    metrics["nees"] = nees.detach()
    metrics["nees_dof"] = dof_tensor
    metrics["nees_per_dof"] = (nees.detach() / dof_tensor).detach()


def _reset_peak_memory(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(_cuda_device_index(device))


def _write_peak_memory_summary(outputdir: str, stage: str, device: str) -> None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    path = os.path.join(outputdir, "memory_summary.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        summary = {}
    summary[stage] = {
        "max_memory_allocated_mb": torch.cuda.max_memory_allocated(_cuda_device_index(device)) / 1024 ** 2,
        "max_memory_reserved_mb": torch.cuda.max_memory_reserved(_cuda_device_index(device)) / 1024 ** 2,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _cuda_device_index(device: str) -> int:
    index = torch.device(device).index
    return torch.cuda.current_device() if index is None else index


def load_bw_func(network_type: str, checkpoint_path: str, device: str):
    bw_func = bw_func_choose(network_type, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    bw_func.load_state_dict(checkpoint["func_bw_model"])
    return bw_func


def load_ba_func(network_type: str, checkpoint_path: str, device: str):
    ba_func = ba_func_choose(network_type, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ba_func.load_state_dict(checkpoint["func_ba_model"])
    return ba_func


def _sample_batch(
    dataset: BaseDataset,
    config: PlatoTrainConfig,
    vo_by_seq: dict[str, PoseTrajectory] | None,
    step: int | None = None,
):
    if dataset._mode == "train":
        if config.train_sequence_strategy == "cycle":
            index = int(step or 0) % dataset.length()
        elif config.train_sequence_strategy == "random":
            index = torch.randint(0, dataset.length(), (1,)).item()
        else:
            raise ValueError("train_sequence_strategy must be random or cycle")
    else:
        index = 0
    return _get_batch_for_index(dataset, index, config, vo_by_seq)


def _get_batch_for_index(
    dataset: BaseDataset,
    index: int,
    config: PlatoTrainConfig,
    vo_by_seq: dict[str, PoseTrajectory] | None,
):
    if config.supervision == "gt_se23":
        mask_flag = {"EUROC": False, "TUM": True, "Fetch": False, "AquaLux": True}.get(dataset.dataset_name, False)
        return get_gt_se23_supervision_batch(
            dataset,
            index,
            device=config.device,
            window_steps=config.window_steps,
            batch_size=config.batch_size,
            mask_flag=mask_flag,
        )
    if config.supervision == "vo_rate_gt_se23":
        if vo_by_seq is None:
            raise ValueError(
                "vo_by_seq is required for vo_rate_gt_se23 timing support"
            )
        seq = dataset.sequences[index]
        mask_flag = {
            "EUROC": False,
            "TUM": True,
            "Fetch": False,
            "AquaLux": True,
        }.get(dataset.dataset_name, False)
        return get_vo_rate_gt_se23_supervision_batch(
            dataset,
            index,
            vo_by_seq[seq],
            device=config.device,
            window_steps=config.window_steps,
            sup_stride=config.sup_stride,
            sup_rate_hz=config.sup_rate_hz,
            batch_size=config.batch_size,
            mask_flag=mask_flag,
        )
    if config.supervision in ("vo_pose", "vo_se23"):
        mask_flag = False
        if vo_by_seq is None:
            raise ValueError(f"vo_by_seq is required for {config.supervision} supervision")
        seq = dataset.sequences[index]
        batch_fn = (
            get_vo_se23_supervision_batch
            if config.supervision == "vo_se23"
            else get_vo_pose_supervision_batch
        )
        return batch_fn(
            dataset,
            index,
            vo_by_seq[seq],
            device=config.device,
            window_steps=config.window_steps,
            sup_stride=config.sup_stride,
            sup_rate_hz=config.sup_rate_hz,
            batch_size=config.batch_size,
            mask_flag=mask_flag,
            vo_init_mode=config.vo_init_mode,
        )
    raise ValueError(
        "supervision must be one of: gt_se23, vo_rate_gt_se23, "
        "vo_pose, vo_se23"
    )


def _integrate_batch(model, dataset: BaseDataset, batch, config: PlatoTrainConfig):
    y0_batch, R0_batch = dataset.construct_init_forodeint(
        batch.t_state, batch.X_state, batch.bwa_state
    )
    spline = Interpolation.CubicHermiteSpline(batch.spline_time, batch.coeff, device=config.device)
    model.set_u_func(spline.evaluate)
    model.set_u_dot_func(spline.derivative)
    model.set_R0(R0_batch)
    if hasattr(model, "set_imu_window_context"):
        model.set_imu_window_context(
            float(dataset.dt),
            t_min=batch.spline_time[0],
            t_max=batch.spline_time[-1],
        )
    if model.biasfunc_a is not None and hasattr(model.biasfunc_a, "set_initial_ba"):
        model.biasfunc_a.set_initial_ba(y0_batch[..., 13:16])
    if getattr(config, "mean_discretization", "legacy_euler") == "paper_zoh":
        return _integrate_batch_paper_zoh(
            model,
            y0_batch,
            R0_batch,
            batch.t_odeint,
        )
    return odeint_SO3(
        model,
        y0_batch,
        R0_batch,
        batch.t_odeint,
        rtol=1e-7,
        atol=1e-9,
        method=config.integral_method,
    )


def _invariant_transition_matrices_for_rollout(
    model,
    solution: torch.Tensor,
    R_sol: torch.Tensor,
    t_odeint: torch.Tensor,
    config: PlatoTrainConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the outer transition/input pair for the configured rollout.

    ``paper_zoh`` is solver-exact. ``legacy_euler`` deliberately preserves the
    historical first-order outer model for reproducibility.
    """
    solution = solution.detach()
    R_sol = R_sol.detach()
    kwargs = {}
    if getattr(config, "mean_discretization", "legacy_euler") == "paper_zoh":
        corrected_omega, corrected_specific_force = _corrected_imu_sequence(
            model, solution, R_sol
        )
        kwargs = {
            "discretization": "paper_zoh",
            "corrected_omega": corrected_omega,
            "corrected_specific_force": corrected_specific_force,
        }
    else:
        kwargs = {"discretization": "paper_first_order"}

    return invariant_transition_matrices(
        R_sol,
        solution[..., 7:10],
        solution[..., 10:13],
        t=t_odeint.detach(),
        gravity=model.g_const,
        **kwargs,
    )


def _corrected_imu_sequence(
    model,
    solution: torch.Tensor,
    R_sol: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-evaluate the left-endpoint corrected inputs used by a rollout."""
    corrected_omega = []
    corrected_specific_force = []
    with torch.no_grad():
        for step in range(solution.shape[0] - 1):
            y_step = torch.cat(
                [torch.zeros_like(solution[step, ..., :3]), solution[step, ..., 3:]],
                dim=-1,
            )
            omega_step, force_step, _, _ = _evaluate_corrected_imu_and_bias_rates(
                model, y_step, R_sol[step]
            )
            corrected_omega.append(omega_step)
            corrected_specific_force.append(force_step)
    return torch.stack(corrected_omega, dim=0), torch.stack(
        corrected_specific_force, dim=0
    )


def _mse_loss_for_stage(batch, solution, R_sol, config: PlatoTrainConfig, stage: str):
    sup = batch.supervised_indices
    if config.supervision in (*GT_FULL_STATE_SUPERVISION, "vo_se23"):
        residual = _se23_residual_for_config(batch, solution, R_sol, config)
        blocks = _mse_blocks_for_stage(stage, full_state=True, config=config)
        rot_weight, vel_weight, pos_weight = _mse_weights_for_stage(stage, config)
        if stage == "gyro" and getattr(config, "mse_gyro_objective", "se23") == "legacy_so3":
            # BiasDy_loss.SO3log_loss uses torch.nn.MSELoss on the three
            # logarithm coordinates.  The invariant left/right residuals have
            # the same squared norm, so this exactly preserves the baseline
            # scalar objective while retaining the right-invariant adjoint.
            loss = 1e6 * (float(rot_weight) * residual[..., :3]).square().mean()
        else:
            loss = 1e6 * se23_mse_loss(
                residual,
                rot_weight=rot_weight,
                vel_weight=vel_weight,
                pos_weight=pos_weight,
                blocks=blocks,
            )
        metrics = se23_block_mse(residual)
        return loss, metrics

    residual = _pose_residual_for_config(batch, solution, R_sol, config)
    blocks = _mse_blocks_for_stage(stage, full_state=False, config=config)
    rot_weight, _, pos_weight = _mse_weights_for_stage(stage, config)
    if stage == "gyro" and getattr(config, "mse_gyro_objective", "se23") == "legacy_so3":
        loss = 1e6 * (float(rot_weight) * residual[..., :3]).square().mean()
    else:
        loss = 1e6 * pose_mse_loss(
            residual,
            rot_weight=rot_weight,
            pos_weight=pos_weight,
            blocks=blocks,
        )
    metrics = pose_block_mse(residual)
    return loss, metrics


def _gyro_objective_gradient_scale(config: PlatoTrainConfig) -> float:
    """Map the half-squared SO(3) adjoint to the selected scalar objective."""
    if getattr(config, "mse_gyro_objective", "se23") == "legacy_so3":
        # 0.5 * mean_states(sum_i r_i^2) -> mean_states,i(r_i^2)
        return 2.0 / 3.0
    return 1.0


def _sparse_gt_nll_streaming_covariance_double_adjoint_backward(
    batch,
    solution,
    R_sol,
    solution_nominal,
    R_nominal,
    state_perturbation: torch.Tensor,
    config: PlatoTrainConfig,
    noise_params: PlatoNoiseParams,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Backpropagate sparse GT NLL with interval-streamed covariance gradients.

    The complete NLL first runs with a small leaf for each interval covariance.
    Its backward pass yields the exact ``dL/dQ_i`` together with residual/H and
    noise-parameter gradients. The paper-aligned default explicitly propagates
    each interval's ``dQ_i/dxi_k`` forward, contracts it with ``dL/dQ_i``, and
    releases it before processing the next interval. An analytic reverse VJP
    remains available only as an implementation-validation/efficiency ablation.

    Both modes preserve the derivative through state-dependent ``G_k``, but
    only ``forward_sensitivity`` is the formal training method. ``F_k`` remains
    state-independent, so no ``dF`` term is required.
    """
    if (
        config.supervision not in GT_FULL_STATE_SUPERVISION
        or config.nll_gt_mode != "sparse"
    ):
        raise ValueError("Streaming covariance is defined for sparse GT NLL")
    if config.covariance_gradient_mode != "forward_sensitivity":
        raise ValueError("Streaming covariance requires forward_sensitivity")
    if torch.any(state_perturbation.detach() != 0):
        raise ValueError("State perturbation must be evaluated at zero")

    sup = build_gt_nll_supervised_indices(
        batch.supervised_indices,
        dense_stride=config.nll_gt_sup_stride,
    ).to(device=R_sol.device, dtype=torch.long)
    solution_nll = solution.to(dtype=torch.float64)
    R_sol_nll = R_sol.to(dtype=torch.float64)
    t_nll = batch.t_odeint.to(device=R_sol.device, dtype=torch.float64)

    solution_nominal_nll = solution_nominal.to(
        device=R_sol.device, dtype=torch.float64
    ).detach()
    R_nominal_nll = R_nominal.to(
        device=R_sol.device, dtype=torch.float64
    ).detach()
    v_nominal = solution_nominal_nll[..., 7:10]
    p_nominal = solution_nominal_nll[..., 10:13]

    Qn = noise_params.Qn().to(device=R_sol.device, dtype=torch.float64)
    train_noise = any(parameter.requires_grad for parameter in noise_params.parameters())

    # A bias-only block freezes the noise/covariance parameters. In that case
    # propagate only the nominal Phi/Q values: computing dQ/d(log sigma) would
    # add work and memory unrelated to the bias double-adjoint comparison.
    if train_noise:
        log_sigma = noise_params.log_sigma_vector()
        Qn_sensitivity = noise_params.Qn_log_sigma_sensitivity().to(
            device=R_sol.device, dtype=torch.float64
        )
        if Qn_sensitivity.shape[0] != log_sigma.numel():
            raise RuntimeError(
                "Noise coordinates and Qn sensitivity directions disagree"
            )
        covariance = preintegrate_supervision_covariances_forward_sensitivity(
            R_nominal_nll,
            v_nominal,
            p_nominal,
            t_nll,
            sup,
            Qn.detach(),
            Qn_sensitivity=Qn_sensitivity.detach(),
            include_state_sensitivity=False,
        )
        if covariance.noise is None:
            raise RuntimeError(
                "Noise covariance forward sensitivity was not computed"
            )
        Phi_list = covariance.Phi
        covariance_values = covariance.covariance
    else:
        Phi_list, covariance_values = preintegrate_supervision_covariances(
            R_nominal_nll,
            v_nominal,
            p_nominal,
            t_nll,
            sup,
            Qn.detach(),
        )

    # A compact covariance leaf exposes dL/dQ_i without attaching the nominal
    # state trajectory to reverse-mode autograd.
    covariance_leaf = covariance_values.detach().requires_grad_(True)
    Q_list = covariance_leaf
    if train_noise:
        log_sigma_delta = log_sigma - log_sigma.detach()
        noise_surrogate = torch.tensordot(
            log_sigma_delta.to(device=R_sol.device, dtype=torch.float64),
            covariance.noise,
            dims=([0], [0]),
        )
        Q_list = Q_list + noise_surrogate

    residual = _sparse_gt_nll_residual(
        batch,
        solution_nll,
        R_sol_nll,
        config,
        supervised_indices=sup,
    )
    include_logdet_h = getattr(config, "nll_gt_include_logdet_h", True)
    if include_logdet_h and config.nll_gt_residual_mode != "right":
        raise ValueError(
            "nll_gt_include_logdet_h=True requires the paper-aligned "
            "nll_gt_residual_mode='right'"
        )
    nll_gt_blocks = getattr(config, "nll_gt_blocks", "full")
    if nll_gt_blocks == "rot":
        loss = sparse_rotation_nll(
            residual,
            Phi_list,
            Q_list,
            prior_std=config.prior_std,
            include_prior=False,
            include_logdet=True,
            include_logdet_h=include_logdet_h,
            jitter=config.cov_jitter,
            reduction="mean",
            time_first=True,
        )
    elif nll_gt_blocks == "full":
        loss = sparse_full_state_nll(
            residual,
            Phi_list,
            Q_list,
            prior_std=config.prior_std,
            include_prior=False,
            include_logdet=True,
            include_logdet_h=include_logdet_h,
            jitter=config.cov_jitter,
            reduction="mean",
            time_first=True,
        )
    else:
        raise ValueError("nll_gt_blocks must be full or rot")

    # Metrics are diagnostics only. Evaluating NEES without a graph avoids
    # retaining a second Cholesky graph alongside the loss graph.
    metrics = _detach_metrics(se23_block_mse(residual))
    metrics["nll"] = loss.detach()
    if config.nll_gt_residual_mode == "right":
        logdet_fn = (
            so3_right_jacobian_inverse_logabsdet
            if nll_gt_blocks == "rot"
            else se23_right_jacobian_inverse_logabsdet
        )
        logdet_residual = residual[..., :3] if nll_gt_blocks == "rot" else residual
        metrics["logdet_h"] = logdet_fn(logdet_residual).sum(dim=0).mean().detach()
    with torch.no_grad():
        nees_fn = sparse_rotation_nees if nll_gt_blocks == "rot" else sparse_full_state_nees
        nees = nees_fn(
            residual.detach(),
            Phi_list.detach(),
            Q_list.detach(),
            prior_std=config.prior_std,
            include_prior=False,
            jitter=config.cov_jitter,
            reduction="mean",
            time_first=True,
        )
    nees_state_dim = 3 if nll_gt_blocks == "rot" else 9
    _add_nees_metrics(metrics, nees, nees_state_dim * int(Phi_list.shape[0]))

    if config.nll_mse_anchor_weight > 0:
        anchor_residual = _se23_residual_for_config(batch, solution, R_sol, config)
        anchor = 1e6 * se23_mse_loss(
            anchor_residual,
            rot_weight=config.mse_rot_weight,
            vel_weight=config.mse_vel_weight,
            pos_weight=config.mse_pos_weight,
            blocks=("rot", "vel", "pos"),
        )
        metrics["mse_anchor"] = anchor.detach()
        loss = loss + config.nll_mse_anchor_weight * anchor

    loss.backward()
    if state_perturbation.grad is None:
        raise RuntimeError("Sparse NLL failed to produce a state gradient")
    if covariance_leaf.grad is None:
        raise RuntimeError("Sparse NLL failed to produce dL/dQ")
    state_gradient = state_perturbation.grad.detach().clone()
    covariance_gradient = covariance_leaf.grad.detach()

    state_gradient_mode = getattr(
        config,
        "nll_covariance_state_gradient_mode",
        "forward_sensitivity",
    )
    if state_gradient_mode not in ("analytic_vjp", "forward_sensitivity"):
        raise ValueError(
            "nll_covariance_state_gradient_mode must be analytic_vjp or "
            "forward_sensitivity"
        )
    state_chunk_size = int(
        getattr(config, "nll_covariance_state_chunk_size", 8)
    )
    if state_chunk_size < 1:
        raise ValueError("nll_covariance_state_chunk_size must be positive")

    # Consume one interval at a time. The formal path explicitly propagates
    # the covariance state sensitivity forward. The analytic VJP branch is an
    # optional equivalence/efficiency ablation and is not the paper method.
    for interval in range(sup.numel() - 1):
        start_idx = int(sup[interval].item())
        end_idx = int(sup[interval + 1].item())
        if state_gradient_mode == "analytic_vjp":
            interval_gradient = preintegrate_interval_covariance_state_vjp(
                R_nominal_nll,
                v_nominal,
                p_nominal,
                t_nll,
                start_idx,
                end_idx,
                Qn.detach(),
                covariance_gradient[interval],
            )
        else:
            interval_gradient = (
                contract_interval_covariance_state_forward_sensitivity_chunked(
                    R_nominal_nll,
                    v_nominal,
                    p_nominal,
                    t_nll,
                    start_idx,
                    end_idx,
                    Qn.detach(),
                    covariance_gradient[interval],
                    chunk_size=state_chunk_size,
                )
            )
        state_gradient[start_idx:end_idx].add_(
            interval_gradient.to(
                device=state_gradient.device, dtype=state_gradient.dtype
            )
        )
        del interval_gradient

    return loss, metrics, state_gradient


def _preintegrate_nll_covariances(
    R: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    t: torch.Tensor,
    supervised_indices: torch.Tensor,
    noise_params: PlatoNoiseParams,
    config: PlatoTrainConfig,
    *,
    covariance_nominal: tuple[torch.Tensor, torch.Tensor] | None = None,
    covariance_state_perturbation: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build NLL covariance with either reverse AD or explicit sensitivity.

    In ``forward_sensitivity`` mode, the noise gradient follows the paper's
    Eq. 12 recursion and is attached to the NLL through a zero-valued linear
    surrogate. For a double-adjoint state update, the same construction adds
    the covariance contribution to the local invariant state gradient. Here
    ``F_k`` has no state derivative; all state dependence is carried by the
    explicit ``dG_k`` terms.

    In an ordinary autograd state update, reverse AD is retained only for the
    low-level ``G(R,v,p)`` state path, while the learned noise coordinates use
    forward sensitivity. ``autograd`` mode preserves the previous all-reverse
    implementation as an ablation/reference path.
    """
    gradient_mode = getattr(
        config, "covariance_gradient_mode", "forward_sensitivity"
    )
    Qn = noise_params.Qn().to(device=R.device, dtype=R.dtype)
    if not torch.is_grad_enabled():
        return preintegrate_supervision_covariances(
            R,
            v,
            p,
            t,
            supervised_indices,
            Qn,
        )
    if gradient_mode == "autograd":
        return preintegrate_supervision_covariances(
            R,
            v,
            p,
            t,
            supervised_indices,
            Qn,
        )
    if gradient_mode != "forward_sensitivity":
        raise ValueError(
            "covariance_gradient_mode must be forward_sensitivity or autograd"
        )

    train_noise = any(parameter.requires_grad for parameter in noise_params.parameters())
    use_state_forward_sensitivity = covariance_state_perturbation is not None
    if not train_noise and not use_state_forward_sensitivity:
        # Bias-only direct autograd still differentiates Q through the nominal
        # state-dependent G_k path, but has no need to build noise-parameter
        # sensitivities for frozen sigma_a/sigma_g.
        return preintegrate_supervision_covariances(
            R,
            v,
            p,
            t,
            supervised_indices,
            Qn.detach(),
        )

    log_sigma = None
    Qn_sensitivity = None
    if train_noise:
        log_sigma = noise_params.log_sigma_vector()
        Qn_sensitivity = noise_params.Qn_log_sigma_sensitivity().to(
            device=R.device, dtype=R.dtype
        )
        if Qn_sensitivity.shape[0] != log_sigma.numel():
            raise RuntimeError(
                "Noise coordinates and Qn sensitivity directions disagree"
            )

    if use_state_forward_sensitivity:
        if covariance_nominal is None:
            raise ValueError(
                "covariance_nominal is required with covariance_state_perturbation"
            )
        solution_nominal, R_nominal = covariance_nominal
        R_for_sensitivity = R_nominal.to(device=R.device, dtype=R.dtype).detach()
        solution_for_sensitivity = solution_nominal.to(
            device=R.device, dtype=R.dtype
        ).detach()
        v_for_sensitivity = solution_for_sensitivity[..., 7:10]
        p_for_sensitivity = solution_for_sensitivity[..., 10:13]
        sensitivity = preintegrate_supervision_covariances_forward_sensitivity(
            R_for_sensitivity,
            v_for_sensitivity,
            p_for_sensitivity,
            t,
            supervised_indices,
            Qn.detach(),
            Qn_sensitivity=(
                None if Qn_sensitivity is None else Qn_sensitivity.detach()
            ),
            include_state_sensitivity=True,
        )
        Phi_list = sensitivity.Phi
        Q_list = sensitivity.covariance
    else:
        # Preserve the exact model-state derivative through state-dependent G,
        # but detach Qn so the covariance rollout does not create a reverse
        # graph for the learned noise parameters.
        Phi_list, Q_list = preintegrate_supervision_covariances(
            R,
            v,
            p,
            t,
            supervised_indices,
            Qn.detach(),
        )
        sensitivity = preintegrate_supervision_covariances_forward_sensitivity(
            R.detach(),
            v.detach(),
            p.detach(),
            t,
            supervised_indices,
            Qn.detach(),
            Qn_sensitivity=Qn_sensitivity.detach(),
            include_state_sensitivity=False,
        )

    if train_noise:
        if sensitivity.noise is None:
            raise RuntimeError("Noise covariance forward sensitivity was not computed")
        log_sigma_delta = log_sigma - log_sigma.detach()
        noise_surrogate = torch.tensordot(
            log_sigma_delta.to(device=R.device, dtype=R.dtype),
            sensitivity.noise,
            dims=([0], [0]),
        )
        Q_list = Q_list + noise_surrogate

    if use_state_forward_sensitivity:
        if sensitivity.state is None:
            raise RuntimeError("State covariance forward sensitivity was not computed")
        state_delta = covariance_state_perturbation.to(
            device=R.device, dtype=R.dtype
        )
        if state_delta.shape[-1] != 9:
            raise ValueError(
                "covariance_state_perturbation must end in 9 tangent coordinates"
            )
        if torch.any(state_delta.detach() != 0):
            raise ValueError(
                "Forward-sensitivity state surrogate must be evaluated at zero perturbation"
            )
        state_surrogates = []
        supervised_indices = supervised_indices.to(
            device=state_delta.device, dtype=torch.long
        )
        for interval, interval_sensitivity in enumerate(sensitivity.state):
            start_idx = int(supervised_indices[interval].item())
            end_idx = int(supervised_indices[interval + 1].item())
            interval_delta = state_delta[start_idx:end_idx].movedim(-1, 1)
            while interval_delta.ndim < interval_sensitivity.ndim - 2:
                interval_delta = interval_delta.unsqueeze(-1)
            interval_delta = interval_delta.unsqueeze(-1).unsqueeze(-1)
            state_surrogates.append(
                (interval_sensitivity * interval_delta).sum(dim=(0, 1))
            )
        Q_list = Q_list + torch.stack(state_surrogates, dim=0)

    return Phi_list, Q_list


def _nll_loss_for_batch(
    batch,
    solution,
    R_sol,
    config: PlatoTrainConfig,
    noise_params,
    *,
    covariance_nominal: tuple[torch.Tensor, torch.Tensor] | None = None,
    covariance_state_perturbation: torch.Tensor | None = None,
):
    sup = batch.supervised_indices
    if config.supervision in GT_FULL_STATE_SUPERVISION:
        sup = build_gt_nll_supervised_indices(
            sup,
            dense_stride=config.nll_gt_sup_stride,
        )

    # Keep the network/ODE forward pass in its native dtype, but evaluate all
    # covariance propagation, Lie residuals, factorizations, and log-determinants
    # for the NLL in float64.  Tensor casts preserve autograd back to float32
    # model parameters.
    solution_nll = solution.to(dtype=torch.float64)
    R_sol_nll = R_sol.to(dtype=torch.float64)
    t_nll = batch.t_odeint.to(device=R_sol.device, dtype=torch.float64)
    Phi_list, Q_list = _preintegrate_nll_covariances(
        R_sol_nll,
        solution_nll[..., 7:10],
        solution_nll[..., 10:13],
        t_nll,
        sup,
        noise_params,
        config,
        covariance_nominal=covariance_nominal,
        covariance_state_perturbation=covariance_state_perturbation,
    )
    if config.supervision in GT_FULL_STATE_SUPERVISION:
        if config.nll_gt_mode == "sparse":
            residual = _sparse_gt_nll_residual(
                batch,
                solution_nll,
                R_sol_nll,
                config,
                supervised_indices=sup,
            )
            include_logdet_h = getattr(
                config, "nll_gt_include_logdet_h", True
            )
            if include_logdet_h and config.nll_gt_residual_mode != "right":
                raise ValueError(
                    "nll_gt_include_logdet_h=True requires the paper-aligned "
                    "nll_gt_residual_mode='right'"
                )
            nll_gt_blocks = getattr(config, "nll_gt_blocks", "full")
            if nll_gt_blocks == "rot":
                loss_fn = sparse_rotation_nll
                nees_fn = sparse_rotation_nees
                nees_state_dim = 3
            elif nll_gt_blocks == "full":
                loss_fn = sparse_full_state_nll
                nees_fn = sparse_full_state_nees
                nees_state_dim = 9
            else:
                raise ValueError("nll_gt_blocks must be full or rot")
            loss = loss_fn(
                residual,
                Phi_list,
                Q_list,
                prior_std=config.prior_std,
                include_prior=False,
                include_logdet=True,
                include_logdet_h=include_logdet_h,
                jitter=config.cov_jitter,
                reduction="mean",
                time_first=True,
            )
            nees = nees_fn(
                residual,
                Phi_list,
                Q_list,
                prior_std=config.prior_std,
                include_prior=False,
                jitter=config.cov_jitter,
                reduction="mean",
                time_first=True,
            )
            nees_dof = nees_state_dim * int(Phi_list.shape[0])
        elif config.nll_gt_mode == "woodbury":
            if getattr(config, "nll_gt_blocks", "full") != "full":
                raise ValueError("rotation-marginal GT NLL requires sparse mode")
            X_ref = _se23_ref_at_supervision(
                batch,
                supervised_indices=sup,
            ).to(device=R_sol.device, dtype=torch.float64)
            residual = se23_log_residual(
                X_ref,
                R_sol_nll[sup],
                solution_nll[sup],
            )
            Lambda = build_se23_precision(
                Phi_list, Q_list, prior_std=config.prior_std, jitter=config.cov_jitter
            )
            loss = full_state_nll(
                residual,
                Lambda,
                meas_sigma_rot=config.meas_sigma_rot_gt,
                meas_sigma_vel=config.meas_sigma_vel_gt,
                meas_sigma_pos=config.meas_sigma_pos_gt,
                jitter=config.cov_jitter,
                reduction="mean",
                time_first=True,
            )
            nees = full_state_nees(
                residual,
                Lambda,
                meas_sigma_rot=config.meas_sigma_rot_gt,
                meas_sigma_vel=config.meas_sigma_vel_gt,
                meas_sigma_pos=config.meas_sigma_pos_gt,
                jitter=config.cov_jitter,
                reduction="mean",
                time_first=True,
            )
            nees_dof = 9 * int(residual.shape[0])
        else:
            raise ValueError("nll_gt_mode must be woodbury or sparse")
        metrics = se23_block_mse(residual)
        metrics["nll"] = loss.detach()
        if config.nll_gt_mode == "sparse" and config.nll_gt_residual_mode == "right":
            nll_gt_blocks = getattr(config, "nll_gt_blocks", "full")
            logdet_fn = (
                so3_right_jacobian_inverse_logabsdet
                if nll_gt_blocks == "rot"
                else se23_right_jacobian_inverse_logabsdet
            )
            logdet_residual = residual[..., :3] if nll_gt_blocks == "rot" else residual
            metrics["logdet_h"] = logdet_fn(logdet_residual).sum(dim=0).mean().detach()
        _add_nees_metrics(metrics, nees, nees_dof)
        if config.nll_mse_anchor_weight > 0:
            anchor_residual = _se23_residual_for_config(batch, solution, R_sol, config)
            anchor = 1e6 * se23_mse_loss(
                anchor_residual,
                rot_weight=config.mse_rot_weight,
                vel_weight=config.mse_vel_weight,
                pos_weight=config.mse_pos_weight,
                blocks=("rot", "vel", "pos"),
            )
            metrics["mse_anchor"] = anchor.detach()
            loss = loss + config.nll_mse_anchor_weight * anchor
        return loss, metrics

    residual = _pose_residual_for_config(batch, solution_nll, R_sol_nll, config)
    if config.vo_residual_mode == "relative":
        if Phi_list.shape[0] < 2:
            raise ValueError("relative VO NLL needs at least three supervised states")
        P0 = Q_list[0]
        Phi_for_nll = Phi_list[1:]
        Q_for_nll = Q_list[1:]
    else:
        P0 = None
        Phi_for_nll = Phi_list
        Q_for_nll = Q_list
    if config.nll_vo_mode == "sparse":
        loss = sparse_pose_nll(
            residual,
            Phi_for_nll,
            Q_for_nll,
            meas_sigma_rot=config.meas_sigma_rot_vo,
            meas_sigma_pos=config.meas_sigma_pos_vo,
            prior_std=config.prior_std,
            P0=P0,
            include_logdet=True,
            jitter=config.cov_jitter,
            reduction="mean",
            time_first=True,
        )
        nees = sparse_pose_nees(
            residual,
            Phi_for_nll,
            Q_for_nll,
            meas_sigma_rot=config.meas_sigma_rot_vo,
            meas_sigma_pos=config.meas_sigma_pos_vo,
            prior_std=config.prior_std,
            P0=P0,
            jitter=config.cov_jitter,
            reduction="mean",
            time_first=True,
        )
    elif config.nll_vo_mode == "woodbury":
        Lambda = build_se23_precision(
            Phi_for_nll, Q_for_nll, prior_std=config.prior_std, jitter=config.cov_jitter
        )
        loss = pose_nll(
            residual,
            Lambda,
            meas_sigma_rot=config.meas_sigma_rot_vo,
            meas_sigma_pos=config.meas_sigma_pos_vo,
            jitter=config.cov_jitter,
            reduction="mean",
            time_first=True,
        )
        nees = pose_nees(
            residual,
            Lambda,
            meas_sigma_rot=config.meas_sigma_rot_vo,
            meas_sigma_pos=config.meas_sigma_pos_vo,
            jitter=config.cov_jitter,
            reduction="mean",
            time_first=True,
        )
    else:
        raise ValueError("nll_vo_mode must be woodbury or sparse")
    metrics = pose_block_mse(residual)
    metrics["nll"] = loss.detach()
    _add_nees_metrics(metrics, nees, 6 * int(residual.shape[0]))
    if config.nll_mse_anchor_weight > 0:
        anchor_residual = _pose_residual_for_config(batch, solution, R_sol, config)
        anchor = 1e6 * pose_mse_loss(
            anchor_residual,
            rot_weight=config.mse_rot_weight,
            pos_weight=config.mse_pos_weight,
            blocks=("rot", "pos"),
        )
        metrics["mse_anchor"] = anchor.detach()
        loss = loss + config.nll_mse_anchor_weight * anchor
    return loss, metrics


def _sparse_gt_nll_residual(
    batch,
    solution,
    R_sol,
    config: PlatoTrainConfig,
    supervised_indices: torch.Tensor | None = None,
):
    sup = batch.supervised_indices if supervised_indices is None else supervised_indices
    X_ref = _se23_ref_at_supervision(
        batch,
        supervised_indices=sup,
    ).to(device=R_sol.device, dtype=R_sol.dtype)
    if config.nll_gt_residual_mode == "right":
        return se23_gt_right_residual(X_ref, R_sol[sup], solution[sup])
    if config.nll_gt_residual_mode == "log":
        return se23_log_residual(X_ref, R_sol[sup], solution[sup])
    if config.nll_gt_residual_mode == "hybrid_global":
        right_residual = se23_gt_right_residual(X_ref, R_sol[sup], solution[sup])
        global_residual = se23_global_residual(X_ref, R_sol[sup], solution[sup])
        return torch.cat((right_residual[..., :3], -global_residual[..., 3:]), dim=-1)
    raise ValueError("nll_gt_residual_mode must be right, log, or hybrid_global")


def _se23_residual_for_config(batch, solution, R_sol, config: PlatoTrainConfig):
    sup = batch.supervised_indices
    X_ref = _se23_ref_at_supervision(batch).to(device=R_sol.device, dtype=R_sol.dtype)
    if config.supervision == "vo_se23" and config.vo_residual_mode == "relative":
        return relative_se23_residual(X_ref, R_sol[sup], solution[sup], mode=config.mse_residual_mode)
    if config.supervision == "vo_se23" and config.vo_residual_mode != "absolute":
        raise ValueError("vo_residual_mode must be absolute or relative")
    if config.mse_residual_mode == "log":
        return se23_log_residual(X_ref, R_sol[sup], solution[sup])
    if config.mse_residual_mode == "right":
        return se23_gt_right_residual(X_ref, R_sol[sup], solution[sup])
    if config.mse_residual_mode == "global":
        return se23_global_residual(X_ref, R_sol[sup], solution[sup])
    if config.mse_residual_mode == "hybrid":
        log_residual = se23_log_residual(X_ref, R_sol[sup], solution[sup])
        global_residual = se23_global_residual(X_ref, R_sol[sup], solution[sup])
        return torch.cat((log_residual[..., :3], global_residual[..., 3:]), dim=-1)
    raise ValueError("mse_residual_mode must be log, right, global, or hybrid")


def _se23_ref_at_supervision(
    batch,
    supervised_indices: torch.Tensor | None = None,
):
    original_sup = batch.supervised_indices
    sup = original_sup if supervised_indices is None else supervised_indices
    if batch.X_ref is None:
        raise ValueError("SE23 supervision requires X_ref")

    # Dense GT references are state-aligned.  VO SE_2(3) references, by
    # contrast, are stored only at the original supervision indices.
    if batch.X_ref.shape[0] == batch.t_state.shape[0]:
        return batch.X_ref[sup]
    if batch.X_ref.shape[0] == original_sup.shape[0]:
        if torch.equal(sup, original_sup):
            return batch.X_ref
        positions = torch.searchsorted(original_sup, sup)
        if torch.any(positions >= original_sup.numel()) or not torch.equal(
            original_sup[positions], sup
        ):
            raise ValueError("Requested SE23 reference is not in the supervision set")
        return batch.X_ref[positions]
    if batch.X_ref.shape[0] == sup.shape[0]:
        return batch.X_ref
    return batch.X_ref[sup]


def _pose_residual_for_config(batch, solution, R_sol, config: PlatoTrainConfig):
    sup = batch.supervised_indices
    R_ref = batch.R_ref.to(device=R_sol.device, dtype=R_sol.dtype)
    p_ref = batch.p_ref.to(device=solution.device, dtype=solution.dtype)
    if config.vo_residual_mode == "relative":
        return relative_pose_residual(
            R_ref,
            p_ref,
            R_sol[sup],
            solution[sup, ..., 10:13],
            mode=config.mse_residual_mode,
        )
    if config.vo_residual_mode != "absolute":
        raise ValueError("vo_residual_mode must be absolute or relative")
    if config.mse_residual_mode == "log":
        return se23_pose_residual(
            R_ref, p_ref, R_sol[sup], solution[sup, ..., 10:13]
        )
    if config.mse_residual_mode == "right":
        return pose_right_residual(
            R_ref, p_ref, R_sol[sup], solution[sup, ..., 10:13]
        )
    if config.mse_residual_mode == "global":
        return pose_global_residual(
            R_ref, p_ref, R_sol[sup], solution[sup, ..., 10:13]
        )
    if config.mse_residual_mode == "hybrid":
        log_residual = se23_pose_residual(
            R_ref, p_ref, R_sol[sup], solution[sup, ..., 10:13]
        )
        global_residual = pose_global_residual(
            R_ref, p_ref, R_sol[sup], solution[sup, ..., 10:13]
        )
        return torch.cat((log_residual[..., :3], global_residual[..., 3:]), dim=-1)
    raise ValueError("mse_residual_mode must be log, global, or hybrid")


def _mse_blocks_for_stage(stage: str, full_state: bool, config: PlatoTrainConfig) -> tuple[str, ...]:
    if stage == "gyro":
        if not config.mse_gyro_full_state:
            return ("rot",)
        block_weights = (
            ("rot", config.mse_rot_weight),
            ("vel", config.mse_vel_weight),
            ("pos", config.mse_pos_weight),
        ) if full_state else (
            ("rot", config.mse_rot_weight),
            ("pos", config.mse_pos_weight),
        )
        blocks = tuple(name for name, weight in block_weights if weight != 0)
        if not blocks:
            raise ValueError("At least one gyro SE23 block weight must be nonzero")
        return blocks
    if stage == "acc":
        block_weights = (
            ("rot", config.mse_acc_rot_weight),
            ("vel", config.mse_acc_vel_weight),
            ("pos", config.mse_acc_pos_weight),
        ) if full_state else (
            ("rot", config.mse_acc_rot_weight),
            ("pos", config.mse_acc_pos_weight),
        )
        blocks = tuple(name for name, weight in block_weights if weight != 0)
        if not blocks:
            raise ValueError("At least one acc SE23 block weight must be nonzero")
        return blocks
    if stage == "joint":
        return ("rot", "vel", "pos") if full_state else ("rot", "pos")
    raise ValueError("stage must be gyro, acc, or joint")


def _mse_weights_for_stage(stage: str, config: PlatoTrainConfig) -> tuple[float, float, float]:
    if stage == "acc":
        return config.mse_acc_rot_weight, config.mse_acc_vel_weight, config.mse_acc_pos_weight
    return config.mse_rot_weight, config.mse_vel_weight, config.mse_pos_weight


def _gyro_window_schedule(config: PlatoTrainConfig) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if config.gyro_multi_window_steps:
        window_steps = tuple(int(window) for window in config.gyro_multi_window_steps)
        if any(window < 2 for window in window_steps):
            raise ValueError("All gyro multi-window steps must be at least 2")
        if config.gyro_multi_window_weights:
            window_weights = tuple(float(weight) for weight in config.gyro_multi_window_weights)
            if len(window_weights) != len(window_steps):
                raise ValueError("--gyro_multi_window_weights must match --gyro_multi_window_steps")
        else:
            window_weights = tuple(1.0 for _ in window_steps)
    else:
        window_steps = (int(config.window_steps),)
        window_weights = (1.0,)
    if any(weight <= 0 for weight in window_weights):
        raise ValueError("All gyro multi-window weights must be positive")
    return window_steps, window_weights


def _validate_gyro_mse(model, dataset_val: BaseDataset, config: PlatoTrainConfig, vo_by_seq):
    window_steps, window_weights = _gyro_window_schedule(config)
    total_loss = 0.0
    window_losses: dict[int, float] = {}
    for window, weight in zip(window_steps, window_weights):
        window_config = replace(config, window_steps=window)
        loss = _validate_mse(model, dataset_val, window_config, stage="gyro", vo_by_seq=vo_by_seq)
        window_losses[window] = loss
        total_loss += float(weight) * loss
    return total_loss, window_losses


def _validate_mse(model, dataset_val: BaseDataset, config: PlatoTrainConfig, stage: str, vo_by_seq):
    model.eval()
    loss_total = 0.0
    with torch.no_grad():
        for index in range(dataset_val.length()):
            batch = _get_batch_for_index(dataset_val, index, config, vo_by_seq)
            solution, R_sol = _integrate_batch(model, dataset_val, batch, config)
            loss, _ = _mse_loss_for_stage(batch, solution, R_sol, config, stage=stage)
            loss_total += loss.item()
    return loss_total


def _validate_nll(model, noise_params, dataset_val: BaseDataset, config: PlatoTrainConfig, vo_by_seq):
    model.eval()
    loss_total = 0.0
    metric_totals: dict[str, float] = {}
    metric_count = 0
    with torch.no_grad():
        for index in range(dataset_val.length()):
            batch = _get_batch_for_index(dataset_val, index, config, vo_by_seq)
            solution, R_sol = _integrate_batch(model, dataset_val, batch, config)
            loss, metrics = _nll_loss_for_batch(batch, solution, R_sol, config, noise_params)
            loss_total += loss.item()
            for name, value in metrics.items():
                metric_totals[name] = metric_totals.get(name, 0.0) + _scalar_value(value)
            metric_count += 1
    denom = max(metric_count, 1)
    avg_metrics = {
        name: torch.tensor(total / denom, dtype=torch.float64)
        for name, total in metric_totals.items()
    }
    return loss_total, avg_metrics


def _write_metrics(writer, prefix: str, metrics: dict[str, torch.Tensor], loss, epoch: int):
    writer.add_scalar(f"{prefix}/loss", _scalar_value(loss), epoch)
    _write_scalar_metrics(writer, prefix, metrics, epoch)


def _write_scalar_metrics(writer, prefix: str, metrics: dict[str, torch.Tensor], epoch: int):
    tags = {
        "rot": "se23_mse_rot",
        "vel": "se23_mse_vel",
        "pos": "se23_mse_pos",
        "nll": "nll",
        "nees": "nees",
        "nees_dof": "nees_dof",
        "nees_per_dof": "nees_per_dof",
        "mse_anchor": "mse_anchor",
        "logdet_h": "logdet_h",
    }
    for name, tag in tags.items():
        if name in metrics:
            writer.add_scalar(f"{prefix}/{tag}", _scalar_value(metrics[name]), epoch)


def _scalar_value(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _save_checkpoint(
    path: str,
    config: PlatoTrainConfig,
    code_stage: str,
    epoch: int,
    val_loss: float,
    model,
    optimizer,
    scheduler=None,
    noise_params=None,
    extra_state: dict | None = None,
):
    checkpoint = {
        "metadata": _checkpoint_metadata(config, code_stage),
        "VF_model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "valloss": val_loss,
    }
    if model.biasfunc_w is not None:
        checkpoint["func_bw_model"] = model.biasfunc_w.state_dict()
    if model.biasfunc_a is not None:
        checkpoint["func_ba_model"] = model.biasfunc_a.state_dict()
        if hasattr(model.biasfunc_a, "rate_limit"):
            checkpoint["learned_ba_rate_limit"] = float(
                model.biasfunc_a.rate_limit().detach().cpu()
            )
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if noise_params is not None:
        checkpoint["noise_params"] = noise_params.state_dict()
        sigma_a, sigma_g = noise_params.sigmas()
        checkpoint["learned_imu_covariance"] = {
            "sigma_a": sigma_a.detach().cpu(),
            "sigma_g": sigma_g.detach().cpu(),
            "Qn": noise_params.Qn().detach().cpu(),
        }
    if extra_state is not None:
        checkpoint.update(extra_state)
    torch.save(checkpoint, path)


def _checkpoint_metadata(config: PlatoTrainConfig, code_stage: str) -> dict:
    metadata = asdict(config)
    metadata["code_stage"] = code_stage
    metadata["nll_numerics"] = {
        "dtype": "float64",
        "fixed_jitter": config.cov_jitter,
        "gt_dense_supervision_stride": config.nll_gt_sup_stride,
        "gt_include_logdet_h": config.nll_gt_include_logdet_h,
    }
    metadata["measurement_covariance"] = {
        "gt": {
            "rot": config.meas_sigma_rot_gt,
            "vel": config.meas_sigma_vel_gt,
            "pos": config.meas_sigma_pos_gt,
        },
        "vo": {
            "rot": config.meas_sigma_rot_vo,
            "pos": config.meas_sigma_pos_vo,
        },
    }
    return metadata
