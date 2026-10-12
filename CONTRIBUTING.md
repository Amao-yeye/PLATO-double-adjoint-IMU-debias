# Contributing

Start with the public recipe and ARCHITECTURE.md. Keep kernels, forward rollout, data preparation and optimization separate. Changes to frames, residuals, timestep, sampling, covariance or checkpoint selection need documentation and numerical evidence.

```bash
python -m pip install -e '.[test]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
bash scripts/smoke_test.sh --skip-data
git diff --check
```

Use small synthetic regression fixtures. Gradient changes must compare production backward passes with autograd or finite differences. Report dtype, device and tolerances. Do not change paper metrics or loosen tests to pass. Preserve upstream notices and keep private paths, data and credentials out of the public archive. Select schedules/checkpoints before seeing test metrics; report all test sequences and conditions. Synthetic tests alone do not establish dataset performance or deployment accuracy.
