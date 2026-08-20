"""Presentation layer for the OptionMC Advanced Risk Lab.

The architecture is strict and one-directional:

    numerical engine (src/)  ->  ui adapter (ui/)  ->  Streamlit pages

Nothing in this package implements mathematics. It loads what the pipeline
computed, calls `src/` when a viewer asks for something new, and formats the
result. Keeping the two apart is what makes the numerical work explainable on
its own and the interface replaceable without touching it.
"""
