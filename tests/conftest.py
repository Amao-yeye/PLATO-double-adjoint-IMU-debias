from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "external/Debias_IMU/BiasDy"), str(REPO / "external/Debias_IMU"), str(REPO / "tools")]

import torch
torch.set_num_threads(1)
