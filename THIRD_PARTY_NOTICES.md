# Third-party notices

The root MIT license applies to PLATO's original code. The following upstream
notices must accompany redistribution of the corresponding components.

| Component | Source | License and local notice |
|---|---|---|
| Debias_IMU and SO3diffeq | [UMich-CURLY/Debias_IMU](https://github.com/UMich-CURLY/Debias_IMU) | MIT; `external/Debias_IMU/LICENSE`, copyright 2025 UMich-CURLY |
| Vendored torchdiffeq | [rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq) | MIT; `external/Debias_IMU/Third_party/torchdiffeq/LICENSE`, copyright 2018 Ricky Tian Qi Chen |
| Historical controldiffeq source | [patrick-kidger/NeuralCDE](https://github.com/patrick-kidger/NeuralCDE) | Apache-2.0; `external/Debias_IMU/Third_party/controldiffeq/LICENSE`; unused by supported PLATO workflows and excluded from the Python wheel |

The Debias_IMU snapshot and vendored solvers contain local modifications. Their
presence is not a claim that the snapshot matches an unmodified upstream release.
The release manifest records content hashes for the distributed files.

EuRoC, AquaLux, other datasets, manuscript archives, reference PDFs and figures
have separate distribution terms. They are excluded from the clean source
archive and Python wheel. Obtain the data from its authors; the root code
license grants no dataset or manuscript rights.

OpenVINS support scripts and patches remain in the research workspace. OpenVINS
is an optional external integration and is outside the current acceptance scope.
