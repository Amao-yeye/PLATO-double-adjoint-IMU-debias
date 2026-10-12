import numpy as np
import pytest

from dataset import EUROCDataset
from aqualux_dataset import _data_root_for_module
from pathlib import Path


def write_sequence(root, name):
    t = 1_400_000_000_000_000_000 + np.arange(128, dtype=np.int64) * 5_000_000
    imu = np.column_stack([t, np.zeros((128, 5)), np.full(128, 9.81)])
    gt = np.column_stack([t, np.zeros((128, 3)), np.ones(128), np.zeros((128, 12))])
    for kind, values in (("imu0", imu), ("state_groundtruth_estimate0", gt)):
        path = root / name / "mav0" / kind / "data.csv"
        path.parent.mkdir(parents=True)
        np.savetxt(path, values, delimiter=",", header="fixture", comments="#")


def test_partial_euroc_cache_is_rebuilt_and_unselected_directories_are_ignored(tmp_path):
    data, cache = tmp_path / "data", tmp_path / "cache"
    for name in ("train", "test"):
        write_sequence(data, name)
    (data / "unrelated").mkdir()
    parameters = dict(dataset_name="EUROC", data_dir=str(data), data_cache_dir=str(cache), train_seqs=["train"], val_seqs=["train"], test_seqs=["test"], percent_for_val=0.2, batch_size=2, loss_window=8, dt=0.005, sg_window_bw=1, sg_order_bw=0, sg_window_ba=1, sg_order_ba=0)
    train = EUROCDataset(**parameters, mode="train")
    assert (cache / "test_gt.p").exists()
    (cache / "test_gt.p").unlink()
    test = EUROCDataset(**parameters, mode="test")
    assert (cache / "test_gt.p").exists()
    assert len(test.get_full_trajectory(0)[0]) == 128
    train.dt = 0.01
    with pytest.raises(ValueError, match="timestep"):
        train.get_data_for_biasDy(0)


def test_aqualux_module_finds_data_root_in_checkout_and_shallow_wheel_install(tmp_path):
    assert _data_root_for_module(Path("/tmp/aqualux_dataset.py")) == Path.cwd()
    checkout = tmp_path / "checkout"
    assert _data_root_for_module(checkout / "external/Debias_IMU/BiasDy/aqualux_dataset.py") == checkout
