# AquaLux release validation — 2026-10-12

The stable GT bias-NODE variant was trained from scratch and evaluated on all
five paper test sequences. Both mean metrics improve over the supplied paper's
VO-trained PLATO target. OpenVINS is excluded.

| Metric | Paper VO PLATO | Measured GT variant | Decrease |
|---|---:|---:|---:|
| AOE RMSE (degrees) | 3.9758 | 3.363827 | 15.39% |
| Position RMSE (meters) | 313.0944 | 265.061807 | 15.34% |

The paper's GT-trained HL-CBD means are 4.6806 degrees and 354.8714 m; the
measured variant decreases them by 28.13% and 25.31%. The PLATO comparison has
different supervision, and the HL-CBD comparison has different architecture.
These are numerical comparisons, not exact paper-protocol reproduction.

| Sequence | Samples propagated | Valid metric samples | AOE (degrees) | APE (m) |
|---|---:|---:|---:|---:|
| Test_001 | 72,280 | 72,280 | 3.064765 | 153.854538 |
| Test_005 | 76,115 | 76,115 | 2.807642 | 142.655975 |
| Test_007 | 74,243 | 73,630 | 5.846478 | 395.591705 |
| Test_017 | 87,447 | 87,447 | 2.129166 | 339.994995 |
| Test_019 | 75,593 | 74,962 | 2.971081 | 293.211823 |

Aggregation is the unweighted mean of sequence RMSEs. The requirement applies
to both means; individual sequences need not improve every metric. Test_001
APE is slightly higher than the paper VO value of 152.040 m.

## Training and evaluation conditions

- Eight training sequences, with the final 20% of each reserved for validation.
  All five test sequences are disjoint from training/validation.
- Final-step selection was declared before training: 1500 gyro MSE updates,
  1500 accelerometer MSE updates and 300 alternating NLL updates, seed 3407.
- GT MSE uses every native IMU state, approximately 998.17 Hz, with GT poses
  interpolated onto that grid. GT NLL uses stride 4 and native-rate propagation.
- The contractive architecture and exact conventions are documented in
  [ARCHITECTURE.md](ARCHITECTURE.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
  This replaces the manuscript's original unconstrained gyro dynamics and
  bounded-rate accelerometer dynamics; it is explicitly a variant.
- Integration covers the complete raw IMU/GT overlap. A 0.02-second nearest-GT
  mask excludes large sampling gaps from metrics, while propagation continues.
  No trajectory or scale alignment is fitted.
- Evaluation uses first GT R/v/p and first GT-derived bias. This privileged
  initialization does not establish deployment performance.

The exact data source is the historical `0524/ATO/gt_imu_gravity_aligned.tum`
export. Dataset notes flag unresolved physical clock calibration, GT timestamp
gaps/batches and the Test_005 calibration window. This mask does not repair those
issues or certify reference accuracy. The newer candidate-reference dataset
release changes clocks, masks and layout and is not the source of these results.

The earlier original-architecture native-GT diagnostic did not meet the target
(mean AOE 14.1086 degrees, APE 2120.6509 m). Its trained checkpoints and serial
trajectories remain preserved. The stable architecture was motivated by
validation deterioration and long-horizon bias stability; final model steps
were not selected using test metrics.

[AQUALUX_VALIDATION.json](AQUALUX_VALIDATION.json) records per-sequence metrics,
input/checkpoint/source hashes, environment and numerical comparison. Numerical
tests verify the complete nested GT/VO NLL gradient against autograd on CPU/CUDA,
the contractive Jacobian, native-MSE/stride-4-NLL sampling and end-to-end serial
versus batched inference. Independent reviewer agents were unavailable.

## Released weights

`dist/plato-aqualux-gt-stable-stride4.pt` excludes optimizer state and host paths.
All network parameters, physical-scale buffers and noise tensors match the
evaluated final checkpoint exactly.

```bash
python tools/evaluate_aqualux_release.py \
  --checkpoint /path/to/plato-aqualux-gt-stable-stride4.pt \
  --data-dir /path/to/0524_data --outputdir Results/AquaLux/weight_verification \
  --device cuda
python tools/check_paper_metrics.py \
  --metrics Results/AquaLux/weight_verification/metrics.csv \
  --reference aqualux_vo_preintegration
```
