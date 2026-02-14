"""
Publication-quality visualization utilities.

Submodules:
    style - MATLAB-compatible plot styling and color palettes
    plots - Pareto front, convergence, and comparison plot functions
"""

from .style import COLORS, setup_publication_style
from .plots import (
    plot_pareto_2d,
    plot_pareto_3d,
    plot_controller_comparison,
    plot_convergence,
    plot_monlta_analysis,
)

__all__ = [
    "COLORS",
    "setup_publication_style",
    "plot_pareto_2d",
    "plot_pareto_3d",
    "plot_controller_comparison",
    "plot_convergence",
    "plot_monlta_analysis",
]
