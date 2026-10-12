import json
from pathlib import Path

import pytest
import torch

from train_euroc_release import REPO
from train_aqualux_release import resolve_recipe, PAPER_TEST_SEQUENCES
from evaluate_aqualux_release import sequence_metrics


def test_aqualux_gt_recipe_keeps_native_mse_and_stride_four_nll(tmp_path):
    recipe = json.loads((REPO / "configs/aqualux_gt_release.json").read_text())
    args, config = resolve_recipe(recipe, tmp_path / "data", tmp_path / "run", "cpu")
    assert config.supervision == "gt_se23"
    assert config.sup_stride is None
    assert config.sup_rate_hz is None
    assert config.nll_gt_sup_stride == 4
    assert config.test_seqs == tuple(PAPER_TEST_SEQUENCES)
    assert not set(config.train_seqs) & set(config.test_seqs)
    recipe["parameters"]["plato_supervision"] = "vo_rate_gt_se23"
    with pytest.raises(ValueError, match="native-IMU-rate"):
        resolve_recipe(recipe, tmp_path, tmp_path, "cpu")


def test_aqualux_metric_mask_excludes_gt_gaps_without_truncating_rollout():
    n = 12
    t = torch.arange(n)*0.001
    nearest = torch.zeros(n)
    nearest[-1] = 0.3
    class Dataset:
        sequences = ["Test_001"]
        gt_valid_max_dt = 0.02
        def load_imu(self, _):
            return {"t_imu": t.double(), "gt_nearest_dt": nearest}
    X = torch.eye(5).repeat(n, 1, 1)
    R = torch.eye(3).repeat(n, 1, 1)
    solution = torch.zeros(n, 16)
    solution[-1, 10] = 100
    row = sequence_metrics(Dataset(), 0, {"t_gt": t, "X_gt": X, "R_pred": R, "sol_pred": solution})
    assert row["sample_count"] == n
    assert row["valid_metric_samples"] == n-1
    assert row["position_rmse_m"] == 0
    with pytest.raises(ValueError, match="complete prepared"):
        sequence_metrics(Dataset(), 0, {"t_gt": t[:-1]})
