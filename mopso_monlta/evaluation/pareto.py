"""
Pareto dominance, crowding distance, and Pareto front extraction.

These functions are the mathematical backbone shared by all three
multi-objective optimizers (MOPSO, MONLTA, NTA).

References:
    Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm:
    NSGA-II," IEEE TEVC, 2002.
"""

import numpy as np


def dominates(f1, f2):
    """
    Check Pareto dominance: does f1 dominate f2?

    f1 ≺ f2 iff:
        (1) f1_i ≤ f2_i for ALL objectives i
        (2) f1_j < f2_j for AT LEAST ONE objective j

    All objectives are assumed to be minimized.

    Args:
        f1, f2: objective vectors (lists/arrays of floats)

    Returns:
        True if f1 Pareto-dominates f2
    """
    return (all(a <= b for a, b in zip(f1, f2)) and
            any(a < b for a, b in zip(f1, f2)))


def crowding_distance(objectives):
    """
    Compute crowding distance for each solution in objective space.

    Measures how isolated a solution is on the Pareto front. Solutions
    in sparser regions get higher distances and are preferred during
    archive pruning, promoting a well-spread front.

    Algorithm (NSGA-II):
        For each objective m:
            1. Sort solutions by objective m
            2. Boundary solutions get infinite distance
            3. Interior: distance += (neighbor_above - neighbor_below) / range

    Args:
        objectives: list of objective vectors [[f1, f2, ...], ...]

    Returns:
        list of crowding distances (one per solution)
    """
    n = len(objectives)
    if n <= 2:
        return [float('inf')] * n

    n_obj = len(objectives[0])
    distances = np.zeros(n)

    for m in range(n_obj):
        sorted_idx = np.argsort([obj[m] for obj in objectives])

        distances[sorted_idx[0]] = float('inf')
        distances[sorted_idx[-1]] = float('inf')

        obj_range = objectives[sorted_idx[-1]][m] - objectives[sorted_idx[0]][m]
        if obj_range == 0:
            continue

        for i in range(1, n - 1):
            distances[sorted_idx[i]] += (
                objectives[sorted_idx[i + 1]][m] - objectives[sorted_idx[i - 1]][m]
            ) / obj_range

    return distances.tolist()


def get_pareto_front(solutions, objectives):
    """
    Extract the Pareto-optimal (non-dominated) subset.

    For each solution, check if any other dominates it.
    If not, it belongs to the Pareto front.

    Args:
        solutions: list of decision variable vectors
        objectives: list of objective vectors

    Returns:
        (pareto_solutions, pareto_objectives) — non-dominated subset
    """
    n = len(solutions)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j and dominates(objectives[j], objectives[i]):
                is_pareto[i] = False
                break

    pareto_solutions = [solutions[i] for i in range(n) if is_pareto[i]]
    pareto_objectives = [objectives[i] for i in range(n) if is_pareto[i]]

    return pareto_solutions, pareto_objectives
