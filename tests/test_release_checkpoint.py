import torch

from export_release_checkpoint import export


def test_portable_checkpoint_preserves_parameters_and_removes_host_paths(tmp_path):
    source, output = tmp_path / "training.pt", tmp_path / "inference.pt"
    original = {"func_bw_model": {"weight": torch.tensor([1.0, 2.0])}, "func_ba_model": {"weight": torch.tensor([3.0, 4.0])}, "metadata": {"outputdir": "local-run", "device": "cuda", "mean_discretization": "paper_zoh", "nll_gt_sup_stride": 4}, "optimizer": {"private_training_state": True}}
    torch.save(original, source)
    report = export(source, output)
    restored = torch.load(output, weights_only=True)
    for name in ("func_bw_model", "func_ba_model"):
        torch.testing.assert_close(restored[name]["weight"], original[name]["weight"])
    assert "optimizer" not in restored
    assert "outputdir" not in restored["metadata"]
    assert "device" not in restored["metadata"]
    assert restored["metadata"]["nll_gt_sup_stride"] == 4
    assert restored["source_checkpoint_sha256"] == report["source_checkpoint_sha256"]
