import copy
import json

import pytest

from check_paper_metrics import compare, REFERENCE_PATH
from train_euroc_release import resolve_parameters, REPO
from plato.supervision import get_gt_se23_supervision_batch, build_gt_nll_supervised_indices
from plato.supervision_smoke_test import FakeEurocDataset
import torch


def reference_rows():
    reference = json.loads(REFERENCE_PATH.read_text())["euroc_gt_preintegration"]
    rows = [dict(sequence=name, **values) for name, values in reference["sequences"].items()]
    return rows, reference


def test_metric_gate_requires_both_means_to_improve():
    rows, reference = reference_rows()
    assert not compare(rows, reference)["all_metrics_better"]
    for row in rows:
        row["aoe_deg"] *= 0.9
    assert not compare(rows, reference)["all_metrics_better"]
    for row in rows:
        row["position_rmse_m"] *= 0.9
    assert compare(rows, reference)["all_metrics_better"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_metric_gate_rejects_invalid_values(bad):
    rows, reference = reference_rows()
    rows[0]["position_rmse_m"] = bad
    with pytest.raises(ValueError):
        compare(rows, reference)


def test_metric_gate_rejects_missing_duplicate_or_short_sequences():
    rows, reference = reference_rows()
    for invalid in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError):
            compare(invalid, reference)
    rows[0]["duration_seconds"] -= 1
    with pytest.raises(ValueError, match="Duration mismatch"):
        compare(rows, reference)


def test_public_recipe_activates_supported_configuration(tmp_path):
    recipe = json.loads((REPO / "configs/euroc_release.json").read_text())
    args, config = resolve_parameters(recipe, tmp_path / "output", tmp_path / "data", "cpu")
    assert config.gradient_mode == "double_adjoint"
    assert config.mean_discretization == "paper_zoh"
    assert config.ba_rate_final_limit == 0.005
    assert config.nll_gt_include_logdet_h
    assert config.sup_stride is None
    assert config.nll_gt_sup_stride == 4
    assert set(args.test_seqs).isdisjoint(args.train_seqs)
    assert set(args.test_seqs).isdisjoint(args.val_seqs)
    malformed = copy.deepcopy(recipe)
    malformed["parameters"]["lr_typo"] = 0.1
    with pytest.raises(ValueError, match="Unsupported recipe parameter"):
        resolve_parameters(malformed, tmp_path, tmp_path, "cpu")


def test_input_overlap_policy_checks_coverage_and_preserves_paper_discrepancy():
    rows, reference = reference_rows()
    for row in rows:
        row["duration_seconds"] += 1
        row["input_overlap_duration_seconds"] = row["duration_seconds"]
        row["sample_count"] = row["input_overlap_sample_count"] = 100
    result = compare(rows, reference, duration_policy="input_overlap")
    assert len(result["paper_duration_differences"]) == 5
    rows[0]["sample_count"] = 99
    with pytest.raises(ValueError, match="complete raw"):
        compare(rows, reference, duration_policy="input_overlap")


def test_gt_mse_stays_native_rate_and_nll_retains_every_fourth_state():
    dataset = FakeEurocDataset(batch_size=2)
    batch = get_gt_se23_supervision_batch(dataset, 0, window_steps=64, batch_size=2)
    torch.testing.assert_close(batch.supervised_indices, torch.arange(64))
    torch.testing.assert_close(batch.t_odeint[1:] - batch.t_odeint[:-1], torch.full((63,), dataset.dt))
    nll_indices = build_gt_nll_supervised_indices(batch.supervised_indices, dense_stride=4)
    torch.testing.assert_close(nll_indices, torch.arange(0, 64, 4))
