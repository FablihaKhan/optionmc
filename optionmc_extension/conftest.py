"""Pytest configuration for the OptionMC extension test suite."""
import sys
from pathlib import Path

import matplotlib

# Headless backend: the system Tk install on this machine is broken, and the
# tests must not depend on a display.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")
