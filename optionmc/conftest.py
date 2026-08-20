"""Pytest configuration for the OptionMC test suite.

This file is environment scaffolding only: it adds no test logic and changes no
assertion in tests/. It exists so the existing suite runs deterministically on
machines where matplotlib's default interactive backend is unusable.
"""
import matplotlib

# This must run before anything imports pyplot. The default backend here is
# tkagg, and the CPython 3.12 install ships a broken Tk (tk.tcl is missing), so
# creating a figure manager raises TclError once enough figures accumulate in a
# full-suite run. Agg is headless and writes files, which is all the
# visualization tests actually need.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any figure a test leaves open.

    The visualization tests create figures and never close them, so without this
    the process accumulates open figures across tests, tripping matplotlib's
    20-figure warning and -- with an interactive backend -- real window creation.
    """
    yield
    plt.close("all")
