"""
Evaluation utilities for multi-objective optimization.

Submodules:
    objectives  - Closed-loop simulation and objective evaluation
    pareto      - Dominance, crowding distance, Pareto extraction
    decision    - TOPSIS and compromise selection
"""

from .pareto import dominates, crowding_distance, get_pareto_front
from .objectives import simulate_closed_loop, evaluate_pid_objectives
from .decision import topsis_select, select_best_compromise

__all__ = [
    "dominates",
    "crowding_distance",
    "get_pareto_front",
    "simulate_closed_loop",
    "evaluate_pid_objectives",
    "topsis_select",
    "select_best_compromise",
]
