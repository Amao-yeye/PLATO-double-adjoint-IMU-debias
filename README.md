# PLATO: Double-Adjoint IMU Debiasing

PLATO learns continuous IMU bias dynamics and noise covariance from accurate trajectory observations. Its nested backward pass combines an invariant-error adjoint with a bias-NODE adjoint. This release provides numerical kernels, training code, public EuRoC and AquaLux recipes, and per-sequence evaluation.

The target manuscript is `main_arxiv.tex` in the supplied `PLATO_for_arxiv.zip`. [Architecture and paper correspondence](ARCHITECTURE.md) document conventions and approximations. OpenVINS is outside current acceptance. AquaLux data is supplied separately.

The validated native-MSE / stride-4-NLL model achieves mean AOE 2.247 degrees and APE 622.972 m versus paper GT values 3.076 degrees and 631.279 m. It uses existing MSE initialization and has reported input-duration differences; see [validation evidence and limits](VALIDATION.md).

The from-scratch AquaLux stable GT variant achieves mean AOE 3.364 degrees and APE 265.062 m versus paper VO values 3.976 degrees and 313.094 m, with different supervision. It also improves on the paper's GT-trained HL-CBD baseline. See [AquaLux per-sequence evidence and conditions](AQUALUX_VALIDATION.md).

## Install and verify

Use Python 3.10 or newer. CPU supports tests and small experiments; full training benefits from CUDA. Locally tested: Python 3.11, PyTorch 2.7.0, CUDA 12.8, RTX 5090. Choose a PyTorch build compatible with your GPU before installing.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
bash scripts/smoke_test.sh --skip-data
```

The wheel installs `plato`, upstream runtime modules and `plato-euroc` / `plato-aqualux` entry points. Historical module names preserve checkpoint compatibility. Disabling pytest plugin autoload isolates tests from unrelated system plugins, including ROS. CPU tests need no dataset or ROS.

## Train and evaluate EuRoC

Obtain all eleven sequences from the [official EuRoC dataset page](https://projects.asl.ethz.ch/datasets/euroc-mav/). Each sequence needs `mav0/imu0/data.csv` and `mav0/state_groundtruth_estimate0/data.csv`; images and bags are unnecessary.

```bash
python tools/train_euroc_release.py \
  --config configs/euroc_release.json --data-dir /path/to/EuRoC \
  --outputdir Results/my_euroc_run --device cuda --run-test

python tools/check_paper_metrics.py \
  --metrics Results/my_euroc_run/evaluation/metrics.csv \
  --reference euroc_gt_preintegration --duration-policy input_overlap
```

The recipe trains from scratch with 1801 gyro MSE updates, 1500 accelerometer MSE updates and 300 alternating NLL updates. GT MSE uses native-rate supervision; NLL uses stride 4 while propagating IMU at native rate. The accelerometer NODE has a learnable bounded rate throughout both stages. These are stochastic optimizer updates, not full dataset epochs. Checkpoints use predeclared final-step counts; test metrics never select checkpoints.

Runs save resolved configuration, source/data/checkpoint hashes, environment and per-sequence metrics. The numerical gate requires **both mean AOE and mean APE to improve** on the complete test set and checks integration durations. It fails when the target is unmet. Assumptions and independent evaluation commands are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Train and evaluate AquaLux

Provide the raw `0524_data` directory, containing `0524/<sequence>/imu_diagnostics/imu_left.csv` and `0524/<sequence>/ATO/gt_imu_gravity_aligned.tum`.

```bash
python tools/train_aqualux_release.py \
  --config configs/aqualux_gt_stable.json --data-dir /path/to/0524_data \
  --outputdir Results/AquaLux/my_run --device cuda --run-test

python tools/check_paper_metrics.py \
  --metrics Results/AquaLux/my_run/evaluation/metrics.csv \
  --reference aqualux_vo_preintegration
```

This GT recipe keeps every native IMU state for MSE and uses stride-4 NLL. GT poses are interpolated onto the approximately 998 Hz IMU grid. Its contractive bias NODE is a documented architecture variant; the original architecture is available in `configs/aqualux_gt_release.json`. Five complete held-out trajectories are propagated, and metrics exclude samples farther than 0.02 seconds from an original GT observation. The paper's AquaLux PLATO target uses VO supervision, so a GT run is a numerical comparison under different supervision. Evaluation also reports the paper's GT-trained HL-CBD baseline. See [protocol details](REPRODUCIBILITY.md).

## Reuse the kernels

```python
import torch
from plato.invariant_discretization import se23_zoh_step
R = torch.eye(3, dtype=torch.float64)
z = torch.zeros(3, dtype=torch.float64)
a = torch.tensor([0., 0., 9.81], dtype=torch.float64)
R_next, v_next, p_next, _ = se23_zoh_step(R, z, z, z, a, 0.005)
```

`plato.config` holds configuration; `plato.rollout` is shared by training and evaluation; `plato.covariance` and `plato.nll` handle uncertainty and likelihood. Supported double-adjoint networks use each bias and local IMU input; coupled replacements need extra adjoint terms. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Distribution and attribution

```bash
python tools/build_release.py --output dist/plato-imu-0.1.0.tar.gz
python -m pip wheel --no-deps . --wheel-dir dist
```

The clean archive selects public source, configuration, tests and documentation. Historical experiments, datasets and Git history in the research workspace are excluded from it and the wheel. Original code is MIT-licensed. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [CITATION.cff](CITATION.cff). Datasets, manuscripts and figures have separate distribution terms.
