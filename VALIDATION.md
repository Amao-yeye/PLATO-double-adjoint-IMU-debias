# Release validation — 2026-10-12

The **GT MSE native-rate / NLL stride-4** configuration was evaluated on all
five EuRoC test sequences. OpenVINS is excluded. The from-scratch AquaLux stable
GT variant also improves both means over the paper's VO PLATO target and GT
HL-CBD baseline; [AquaLux evidence](AQUALUX_VALIDATION.md) records its separate
protocol, architecture and data limitations.

| Mean over five sequences | Supplied paper, GT | Measured | Decrease |
|---|---:|---:|---:|
| AOE RMSE (degrees) | 3.0756 | 2.246505 | 26.96% |
| Position RMSE (meters) | 631.2786 | 622.971674 | 1.32% |

Paper averages use the rounded per-sequence table values. Evaluation covers
the complete raw IMU/GT overlap with no trajectory or scale alignment.
[VALIDATION.json](VALIDATION.json) records per-sequence metrics, input durations,
data/checkpoint hashes and the actual environment.

The verified model continues existing native-rate MSE final checkpoints
(1801 gyro updates, 1500 accelerometer updates) with 300 new alternating NLL
updates: 150 bias and 150 covariance updates. NLL uses stride 4; IMU propagation
stays at native rate. This is not a newly completed from-scratch MSE run.
An earlier from-scratch stride-2 diagnostic did not improve both means; no
from-scratch stride-4 performance guarantee has been established.

The paper labels V1_01 as 115.450 s and V2_02 as 144.700 s. Raw input overlap
is respectively 143.555 s and 115.450 s; existing GT-only records give V1_01
as 144.700 s. These differences are reported explicitly, preserving the paper
metrics and labels. This is a numerical comparison over complete input overlap
with an interval difference, rather than exact paper-evaluation reproduction.

Initialization uses first GT rotation, velocity, position and GT-derived bias.
The measurements do not establish online deployment accuracy.

## Verification

Local tests cover sparse/dense likelihoods, gradients, SO(3)/SE₂(3) residuals,
complete GT/noisy-VO nested backward passes on CPU/CUDA, native-MSE/stride-4-NLL
sampling, contractive bias dynamics, serial/batched inference equivalence,
partial caches, metric gates and distribution integrity. The three
existing smoke tests also pass. The installed wheel's smoke tests run outside
the repository. GitHub CPU CI is configured but has not run remotely here.

Independent reviewer agents could not start because their configured model was
unavailable. Evidence consists of numerical tests, direct execution,
archive/installation checks and manual source review.

## Artifacts

Generated files in `dist/`:

- `plato-imu-0.1.0.tar.gz`: clean source, configuration, tests and notices.
- `plato_imu-0.1.0-py3-none-any.whl`: installable runtime.
- `plato-euroc-gt-stride4.pt`: portable inference checkpoint without optimizer
  state or machine paths; network/noise tensors match the evaluated model exactly.
- `euroc-stride4-metrics.csv`: fresh per-sequence metrics.
- `plato-aqualux-gt-stable-stride4.pt`: portable AquaLux inference checkpoint.
- `aqualux-gt-stable-stride4-metrics.csv`: five-sequence AquaLux metrics.

Evaluate the released weights from the extracted source:

```bash
python tools/evaluate_euroc_release.py \
  --checkpoint /path/to/plato-euroc-gt-stride4.pt \
  --data-dir /path/to/EuRoC --outputdir Results/weight_verification --device cuda
python tools/check_paper_metrics.py \
  --metrics Results/weight_verification/metrics.csv \
  --reference euroc_gt_preintegration --duration-policy input_overlap
```

To rerun NLL from compatible native-rate MSE checkpoints, add
`--resume-gyro /path/to/gyro.pt --resume-acc /path/to/acc.pt` to the README
training command. The manifest distinguishes imported stages and new updates.
