# Architecture and paper correspondence

## Module boundaries

```text
configs/euroc_release.json
         |
tools/train_euroc_release.py       tools/evaluate_euroc_release.py
         |                                    |
plato.config - plato.training            learning.Test
                    +-------- plato.rollout --+
                                   |
                    plato.invariant_discretization

loss:          plato.losses + plato.nll + plato.covariance
outer adjoint: plato.analytic_gradient / plato.so3_analytic_gradient
inner adjoint: plato.bias_adjoint / plato.gyro_adjoint
data:          dataset.EUROCDataset / aqualux_dataset.AquaLuxDataset
```

Configuration and forward rollout are separated from optimization. Training and evaluation share `rollout.py`. Compatibility imports in `training.py` retain existing symbols. Validated network names and checkpoint keys are preserved. Public workflows are named in the README; historical experiment scripts are not required by the clean release.

## Conventions

| Quantity | Convention |
|---|---|
| R | Body-to-world |
| SE₂(3) state | Columns `[R, v, p]` |
| Tangent | `[phi, rho_v, rho_p]` |
| Invariant error | `X_true @ inverse(X_nominal)` |
| Perturbation | Left: `Exp(delta) @ X` |
| IMU input | `[gyro, accel]`, rad/s and m/s² |
| Bias/noise Jacobian columns | `[ba, bg]` |
| Runtime vector | `[chart(3), time, bg(3), v(3), p(3), ba(3)]` |
| Gravity | World frame; default `[0, 0, -9.81]` m/s² |

The different IMU and Jacobian coordinate orders are intentional.

## Manuscript mapping

Equation labels refer to `main_arxiv.tex` in the supplied archive.

| Paper component | Production implementation | Numerical contract |
|---|---|---|
| `eq:imukin`, `eq:bias_dynamics` | `rollout.integrate_batch_paper_zoh`, `invariant_discretization.se23_zoh_step` | Exact held-input mean-state propagation; Euler bias integration |
| `eq:IMUXP` | `covariance.se23_phi`, `se23_Gd` | State-independent invariant transition; first-order stochastic noise input |
| `eq:Qis1`, precision | `covariance.preintegrate_supervision_covariances`, `build_se23_precision` | Float64 interval covariance and block-tridiagonal precision |
| `eq:gtnll` | `nll.sparse_full_state_nll` | Quadratic, covariance determinant and `log|det H|` |
| `eq:vonll` | `nll.sparse_pose_nll`, `pose_nll` | Noisy pose likelihood with latent velocity uncertainty |
| Outer invariant adjoint | `analytic_gradient.adjoint_bias_gradient` | Backward recursion with cost jumps and exact ZOH mean-input Jacobian B |
| `eq:inner_adjoint_dynamics` | `bias_adjoint.accumulate_ba_euler_adjoint_gradients`, `gyro_adjoint.accumulate_bg_euler_adjoint_gradients` | Discrete adjoint of executed Euler updates, using local network VJPs |
| Alternating optimization | `training.train_nll_alternating` | Bias phases freeze covariance; covariance phases freeze bias networks |

The manuscript's stochastic G differs from finite-step mean Jacobian B. B retains gyro/acceleration and accelerometer-to-position coupling; they agree to first order in dt. The inner recursion discretizes the continuous derivation to match the actual forward solver.

Sparse/dense likelihood tests compare values and derivatives. ZOH tests check invariance and finite-step adjoints. Complete nested backward tests compare production double-adjoint with full autograd on CPU/CUDA for GT and noisy VO.

## Assumptions and boundaries

Supported NODEs depend on their own bias and local IMU input. The R argument exists for compatibility; canonical networks do not use orientation, velocity, position or the opposite bias. Coupled replacements need additional adjoint paths.

Noisy VO NLL uses general state/covariance derivatives. Streaming intervals are reserved for sparse GT full-state likelihoods. Both paths are tested against autograd. Noise parameters are effective likelihood weights, not established physical calibration. Jitter, clipping and projected bounds are distinct numerical choices. The public recipe changes accelerometer rate parameterization and training budget relative to the paper; report this in comparisons.

## Stable AquaLux variant and inference

`plato.models.BiasRelaxation` has a 6→64→64→3 tanh network of raw IMU values, normalized by serialized physical scales. It predicts a bounded target bias and returns `rate * (target - own_bias)`. Gyro and accelerometer factories use offsets 4 and 13, bounds 0.05 rad/s and 1 m/s², and rate 10 s⁻¹. Targets use no rotation, velocity, position, opposite bias or IMU derivative. Therefore the own-bias Jacobian is exactly -10I and the existing inner adjoint assumptions still apply. All scales are checkpoint buffers. The final target layer starts at zero; subsequent values are learned from training trajectories.

Native Euler updates contract toward the target if rate*dt <= 1. This architecture addresses long-horizon drift and sensitivity to noisy GT-derived initial biases; it is a research variant with its own recipe, not the manuscript's exact network. Numerical stability alone does not establish lower test error.

`plato.inference.rollout_native_sequences` precomputes spline values on the executed floating-point clock, batches independent sequences on matching native grid prefixes and freezes rows after their final state. It calls `plato.rollout` rather than implementing a second integrator. The production serial `learning.Test` remains available. End-to-end CPU/CUDA tests compare trajectories, rotations and corrected IMU under first-GT-bias and zero-bias initialization, including mixed float64 IMU / float32 GT times.
