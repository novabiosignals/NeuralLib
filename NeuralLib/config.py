"""
Default paths when NeuralLib is installed as a pip package.
"""
from pathlib import Path
import os

env_base = os.getenv("NEURALLIB_BASE_DIR")
if env_base is not None:
    BASE_DIR = Path(env_base).expanduser().resolve()
else:
    BASE_DIR = Path.cwd().resolve()

DATA_BASE_DIR = BASE_DIR / "data"
RESULTS_BASE_DIR = BASE_DIR / "results"
HUGGING_MODELS_BASE_DIR = BASE_DIR / "hugging_prodmodels"

for d in (DATA_BASE_DIR, RESULTS_BASE_DIR, HUGGING_MODELS_BASE_DIR):
    d.mkdir(parents=True, exist_ok=True)