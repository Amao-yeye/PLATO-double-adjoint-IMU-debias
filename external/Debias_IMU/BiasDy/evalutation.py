"""Rotation and Euclidean trajectory RMSE with optional validity masks."""
import torch
import lie_algebra as Lie


def ATE(p_gt: torch.Tensor, p_est: torch.Tensor, mask = None) -> torch.Tensor:
    """
    Absolute Trajectory Error
    :param p_gt: ground truth position: torch.tensor (batch_size, 3)
    :param p_est: estimated position: torch.tensor (batch_size, 3)
    :return: ATE: torch.tensor scalar
    """
    assert p_gt.shape == p_est.shape and p_gt.dim() == 2
    if mask is not None:
        assert mask.dim() == 1 and mask.shape[0] == p_gt.shape[0]
        validated_index = (mask == 0) # shape: (batch_size)
        return (p_gt[validated_index] - p_est[validated_index]).square().sum(dim=1).mean().sqrt()
    else:
        return (p_gt - p_est).square().sum(dim=1).mean().sqrt()


def AOE(R_gt: torch.Tensor, R_est: torch.Tensor, mask = None) -> torch.Tensor:
    """
    Absolute Orientation Error
    :param R_gt: ground truth orientation: torch.tensor (batch_size, 3, 3)
    :param R_est: estimated orientation: torch.tensor (batch_size, 3, 3)
    :param mask: mask for the orientation: torch.tensor (batch_size) 0: valid, 1: invalid
    :return: AOE: torch.tensor scalar
    """
    assert R_gt.shape == R_est.shape and R_gt.dim() == 3
    if mask is not None:
        assert mask.dim() == 1 and mask.shape[0] == R_gt.shape[0]
        validated_index = (mask == 0) # shape: (batch_size)
        return Lie.SO3log(R_gt[validated_index] @ R_est[validated_index].transpose(-1, -2)).square().sum(dim=(-1)).mean().sqrt() * 180 / torch.pi
    else:
        return Lie.SO3log(R_gt @ R_est.transpose(-1, -2)).square().sum(dim=(-1)).mean().sqrt() * 180 / torch.pi
