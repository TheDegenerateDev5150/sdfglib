import os
import sys

import pytest
import torch

# Ensure the pytorch project root is importable so that top-level packages such
# as ``tests`` and ``benchmarks`` resolve when this benchmark module is
# collected directly by pytest (e.g. ``pytest benchmarks/torch/model_zoo/...``).
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@pytest.fixture(autouse=True)
def seed_rng():
    """Set a fixed random seed before each test for reproducibility."""
    torch.manual_seed(815)
