"""Validated configuration shared by PLATO training workflows."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlatoTrainConfig:
    supervision: str
    window_steps: int
    sup_stride: int | None
    sup_rate_hz: float | None
    integral_method: str
    batch_size: int
    outputdir: str
    network_type_Gyro: str
    network_type_Acc: str
    meas_sigma_rot_gt: float
    meas_sigma_vel_gt: float
    meas_sigma_pos_gt: float
    meas_sigma_rot_vo: float
    meas_sigma_pos_vo: float
    init_sigma_a: float
    init_sigma_g: float
    plato_noise_isotropic: bool
    nll_min_sigma_a: float | None
    nll_max_sigma_a: float | None
    nll_min_sigma_g: float | None
    nll_max_sigma_g: float | None
    cov_jitter: float
    nll_gt_sup_stride: int
    prior_std: float
    val_freq: int
    save_freq: int
    device: str
    train_sequence_strategy: str = "random"
    mse_rot_weight: float = 1.0
    mse_vel_weight: float = 1.0
    mse_pos_weight: float = 1.0
    mse_acc_rot_weight: float = 0.0
    mse_acc_vel_weight: float = 1.0
    mse_acc_pos_weight: float = 1.0
    mse_gyro_full_state: bool = False
    gyro_train_with_acc: bool = False
    mse_residual_mode: str = "log"
    gyro_multi_window_steps: tuple[int, ...] = ()
    gyro_multi_window_weights: tuple[float, ...] = ()
    nll_gt_mode: str = "sparse"
    nll_gt_blocks: str = "full"
    nll_gt_residual_mode: str = "right"
    nll_gt_include_logdet_h: bool = True
    nll_vo_mode: str = "woodbury"
    nll_mse_anchor_weight: float = 0.0
    vo_residual_mode: str = "absolute"
    vo_init_mode: str = "gt"
    gradient_mode: str = "autograd"
    covariance_gradient_mode: str = "forward_sensitivity"
    nll_stream_covariance_intervals: bool = True
    nll_covariance_state_gradient_mode: str = "forward_sensitivity"
    nll_covariance_state_chunk_size: int = 8
    analytic_grad_scale: float = 1.0
    analytic_grad_clip_norm: float = 0.0
    analytic_residual_jacobian: str = "bch"
    train_seqs: tuple[str, ...] = ()
    val_seqs: tuple[str, ...] = ()
    test_seqs: tuple[str, ...] = ()
    mean_discretization: str = "legacy_euler"
    mse_gyro_objective: str = "se23"
    ba_rate_anneal_start_fraction: float = 0.6
    ba_rate_final_limit: float = 5e-3

    def __post_init__(self) -> None:
        if self.cov_jitter < 0:
            raise ValueError("cov_jitter must be non-negative")
        if self.nll_gt_sup_stride < 2:
            raise ValueError("nll_gt_sup_stride must be at least 2")
        if self.covariance_gradient_mode not in ("forward_sensitivity", "autograd"):
            raise ValueError(
                "covariance_gradient_mode must be forward_sensitivity or autograd"
            )
        if self.nll_covariance_state_gradient_mode not in (
            "analytic_vjp",
            "forward_sensitivity",
        ):
            raise ValueError(
                "nll_covariance_state_gradient_mode must be analytic_vjp or "
                "forward_sensitivity"
            )
        if self.nll_covariance_state_chunk_size < 1:
            raise ValueError("nll_covariance_state_chunk_size must be positive")
        if self.nll_gt_blocks not in ("full", "rot"):
            raise ValueError("nll_gt_blocks must be full or rot")
        if self.nll_gt_blocks == "rot" and self.nll_gt_mode != "sparse":
            raise ValueError("rotation-marginal GT NLL requires nll_gt_mode='sparse'")
        if self.mean_discretization not in ("legacy_euler", "paper_zoh"):
            raise ValueError(
                "mean_discretization must be legacy_euler or paper_zoh"
            )
        if self.mean_discretization == "paper_zoh" and self.integral_method != "euler":
            raise ValueError(
                "paper_zoh uses Euler bias dynamics and requires integral_method='euler'"
            )
        if self.mse_gyro_objective not in ("se23", "legacy_so3"):
            raise ValueError("mse_gyro_objective must be se23 or legacy_so3")
        if self.mse_gyro_objective == "legacy_so3" and self.mse_gyro_full_state:
            raise ValueError(
                "legacy_so3 is a rotation-only objective and is incompatible "
                "with mse_gyro_full_state=True"
            )
        if not 0.0 <= self.ba_rate_anneal_start_fraction < 1.0:
            raise ValueError("ba_rate_anneal_start_fraction must be in [0, 1)")
        if self.ba_rate_final_limit <= 0:
            raise ValueError("ba_rate_final_limit must be positive")
