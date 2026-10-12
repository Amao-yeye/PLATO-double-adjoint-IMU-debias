# Reproducibility and metric acceptance

## Supported workflow

The supported release workflows are EuRoC and AquaLux GT-supervised preintegration: install, verify without data, train from scratch and evaluate all five held-out sequences. Acceptance excludes OpenVINS. VO likelihood gradients are tested synthetically; the public AquaLux GT recipe does not establish VO-trained performance. Historical instructions remain in `docs/research_snapshot_reproducibility.md` in the research checkout.

## Data and split

Download from the [official EuRoC page](https://projects.asl.ethz.ch/datasets/euroc-mav/) and extract into named sequence directories:

```text
EuRoC/MH_01_easy/mav0/imu0/data.csv
EuRoC/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv
... (all eleven sequences)
```

| Purpose | Sequences |
|---|---|
| Train / temporal validation | MH_01_easy, MH_03_medium, MH_05_difficult, V1_02_medium, V2_01_easy, V2_03_difficult |
| Held-out test | MH_02_easy, MH_04_difficult, V1_01_easy, V1_03_difficult, V2_02_medium |

Each training sequence's last 20% is validation; training windows stay in the first 80%. Training is resampled to 200 Hz; test rollouts preserve synchronized timestamps. The adapter uses provided EuRoC velocity labels, whereas the manuscript generally describes deriving velocity from position differences. Exact reproduction of every paper experiment is not claimed.

Research rollouts initialize with the first GT rotation, velocity, position and GT-derived bias. Initial bias uses GT differences over the first IMU interval. This is privileged initialization, not online deployment. The evaluator supports `--bias-init zero` as a different protocol.

Caches live inside each fresh experiment. Reusing caches across changed data, timestep, split or smoothing requires recomputation. The adapter checks every requested IMU/GT cache pair and ignores unrelated directories. Canonical runs require empty output directories.

## Execution and checkpoint selection

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
bash scripts/smoke_test.sh --skip-data
python tools/train_euroc_release.py \
  --config configs/euroc_release.json --data-dir /path/to/EuRoC \
  --outputdir Results/my_run --device cuda --run-test
```

The recipe uses 64 IMU states per dense-GT window, batch size 1000 and seed 3407 per stage. Gyro MSE retains upstream `legacy_so3` normalization. Bias NODEs use Euler; SE₂(3) states use exact constant-input ZOH. GT MSE retains the native IMU grid; NLL retains indices 0, 4, 8, ... without decimating forward propagation. GT NLL factors span four IMU intervals to avoid rank-deficient one-interval full-state process covariances. NLL is float64 with zero fixed jitter and includes the observation-Jacobian determinant. Noise is isotropic within each sensor's three axes.

Requested update counts are recorded before evaluation. The final step of each stage is selected; `best_val_model.pt` is diagnostic. Names are zero-based: 1801 updates produce `model_epoch_1800.pt`. Without `--run-test`, training does not evaluate held-out performance. Independent evaluation:

```bash
python tools/evaluate_euroc_release.py \
  --checkpoint Results/my_run/03_nll/NLL_alternating_weights/model_epoch_299.pt \
  --data-dir /path/to/EuRoC --cache-dir Results/my_run/cache \
  --outputdir Results/my_run/evaluation --device cuda
```

The evaluator checks architecture, discretization and supervision metadata and rejects declared train/validation overlap with test sequences. It saves fresh trajectories and data/cache/checkpoint hashes. Use an empty output folder.

## Metrics

AOE is RMS SO(3) logarithmic rotation error in degrees; APE is RMS Euclidean position error in meters. Pure integration has no trajectory alignment or scale adjustment. Aggregation is the arithmetic mean of five sequence RMSEs, not a pooled frame RMSE.

`configs/paper_metrics.json` transcribes the supplied arXiv tables. EuRoC GT mean targets are approximately 3.076 degrees and 631.279 m. The gate averages the rounded paper rows and requires both means to decrease. The default paper-duration policy rejects duration differences exceeding 0.01 s. The evaluator uses an explicitly reported input-overlap policy: it verifies sample counts and duration against the complete raw IMU/GT overlap and reports differences from the paper. V1_01 and V2_02 have transposed duration labels in the supplied table relative to this data; the V1_01 overlap is 143.555 s rather than the GT-only 144.700 s. Paper metrics are retained verbatim. Both policies reject missing/duplicate/extra sequences and nonfinite/negative metrics.

```bash
python tools/check_paper_metrics.py \
  --metrics Results/my_run/evaluation/metrics.csv \
  --reference euroc_gt_preintegration --duration-policy input_overlap --output Results/my_run/evaluation/acceptance.json
```

Exit 0 means both means improve; exit 1 means at least one does not. Invalid inputs raise errors. The numerical gate does not certify unseen data, initialization or selection history; review provenance too. Historical summaries are not fresh release evidence.

## AquaLux protocol

The raw layout is `0524_data/0524/<sequence>/imu_diagnostics/imu_left.csv` plus `ATO/gt_imu_gravity_aligned.tum`. Source timestamps and input hashes are checked before training. Inputs are read without modifying the dataset. Training uses Train_000, Train_002, Train_008, Train_010, Train_012, Train_014, Train_016 and Train_018, reserving each sequence's last 20% for temporal validation. Test_001, Test_005, Test_007, Test_017 and Test_019 are held out.

The median native IMU interval is approximately 0.001001835 seconds (998.17 Hz). GT MSE uses poses interpolated onto every native IMU state without thinning; GT NLL retains indices 0, 4, 8, ... while propagating every IMU interval. Mocap originally runs at about 120 Hz. Velocity labels use position differences smoothed over 0.12 seconds; gravity is 9.7879 m/s². Noisy GT-derived first-interval biases and first GT R/v/p initialize evaluation. `--bias-init zero` defines a separate protocol.

The stable recipe uses 1500 gyro MSE, 1500 accelerometer MSE and 300 alternating NLL optimizer updates, window length 128, batch size 512 and seed 3407 per stage. Final-step selection is declared before training. `plato.models.BiasRelaxation` models `db/dt = 10 * (L * tanh(h(raw_imu)) - b)` with component limits L=0.05 rad/s for gyro and 1 m/s² for acceleration. At fixed IMU its bias Jacobian is -10I. Native Euler updates are convex combinations when 10*dt <= 1, avoiding unconstrained long-term bias drift. This is a changed NODE architecture, not an exact reproduction of the manuscript architecture; its complete nested NLL gradients are checked against autograd on CPU and CUDA.

Evaluation propagates each complete IMU/GT overlap. AOE/APE use all samples with an original GT observation within 0.02 seconds; IMU propagation continues through GT gaps. Batched independent native-grid inference uses the same ZOH kernel and accumulated floating-point clock as serial evaluation, freezes padding and is checked against the original serial evaluator. `--serial` selects serial inference.

The numerical target is the paper's VO-trained PLATO mean: 3.9758 degrees AOE and 313.0944 m APE. GT-trained results must explicitly report different supervision. A separate GT-trained HL-CBD comparison has targets 4.6806 degrees and 354.8714 m. Neither comparison establishes an exact same-protocol reproduction or deployment performance.

This recipe uses the supplied historical `0524/ATO` export. The separate AquaLux candidate-reference release has shifted clocks, segment-quality masks and a different layout; it must not be silently substituted. Dataset notes flag historical mocap/sensor clock alignment, GT timestamp batches and gaps, and the Test_005 calibration window. The 0.02-second nearest-GT mask removes large sampling gaps, but does not repair clock calibration or certify reference accuracy. Report these results as agreement with the supplied historical labels. The checkpoint and manifests identify the exact input files by hash.

## Portability and distribution

Manifests save configuration, dependency versions, source/data hashes, checkpoint hashes and completion status. Failed runs are marked failed. Load checkpoints only from trusted sources. Hardware/PyTorch differences can change numerical results; exact binary identity across hosts is unnecessary.

The clean archive and wheel exclude data, checkpoints, manuscript archives, private notes and Git history. Previously tracked data remains in the research checkout/history; `.gitignore` cannot remove existing tracked files. The clean archive and wheel are the public distribution boundary.
