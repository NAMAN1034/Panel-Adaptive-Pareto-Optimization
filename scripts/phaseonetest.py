import platform
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())

import numpy, pandas, sklearn, scipy, matplotlib, seaborn, Bio, yaml, tqdm, joblib, pymoo
print(f"numpy {numpy.__version__} | pandas {pandas.__version__} | "
      f"sklearn {sklearn.__version__} | pymoo {pymoo.__version__}")

import torch
print("torch", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

import esm
print("fair-esm imported OK")

from src.utils.seeding import seed_everything
seed_everything(0)
print("Seeding OK. Phase 1 complete.")