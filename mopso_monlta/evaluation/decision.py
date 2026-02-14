"""
Decision-making methods for selecting compromise solutions from Pareto fronts.

Methods:
    topsis_select         - TOPSIS (Hwang & Yoon, 1981) ranking
    select_best_compromise - Euclidean or TOPSIS selection
"""

import numpy as np


def topsis_select(solutions, objectives, weights=None):
    """
    TOPSIS decision-making: select the best compromise from a Pareto front.

    Steps:
        1. Vector-normalize the objective matrix
        2. Apply weights (default: equal)
        3. Find ideal (min) and anti-ideal (max) points
        4. Compute distances d+ (to ideal) and d- (to anti-ideal)
        5. Select by maximum closeness: C = d- / (d+ + d-)

    Args:
        solutions: list of decision variable vectors
        objectives: list of objective vectors
        weights: importance weights (default: equal)

    Returns:
        (best_solution, best_objectives) or (None, None) if empty
    """
    if len(objectives) == 0:
        return None, None

    obj_arr = np.array(objectives)
    n_obj = obj_arr.shape[1]

    if weights is None:
        weights = np.ones(n_obj) / n_obj

    # Normalize and weight
    norms = np.sqrt(np.sum(obj_arr ** 2, axis=0))
    norms[norms == 0] = 1
    norm_obj = obj_arr / norms * weights

    # Ideal and anti-ideal
    ideal = norm_obj.min(axis=0)
    anti_ideal = norm_obj.max(axis=0)

    # Distances
    d_plus = np.sqrt(np.sum((norm_obj - ideal) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((norm_obj - anti_ideal) ** 2, axis=1))

    # Closeness coefficient
    closeness = d_minus / (d_plus + d_minus + 1e-12)
    best_idx = np.argmax(closeness)

    return solutions[best_idx], objectives[best_idx]


def select_best_compromise(solutions, objectives, method='topsis'):
    """
    Select the best compromise solution from a Pareto front.

    Args:
        solutions: list of decision variable vectors
        objectives: list of objective vectors
        method: 'euclidean' or 'topsis'

    Returns:
        (best_solution, best_objectives, best_index)
    """
    if len(objectives) == 0:
        return None, None, -1

    obj_array = np.array(objectives)
    n, m = obj_array.shape

    # Min-max normalization
    obj_min = obj_array.min(axis=0)
    obj_max = obj_array.max(axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1
    obj_norm = (obj_array - obj_min) / obj_range

    if method == 'euclidean':
        # Distance to utopia point (0, 0, ..., 0)
        distances = np.sqrt(np.sum(obj_norm ** 2, axis=1))
        best_idx = np.argmin(distances)

    elif method == 'topsis':
        weights = np.ones(m) / m
        weighted = obj_norm * weights

        PIS = weighted.min(axis=0)
        NIS = weighted.max(axis=0)

        d_pos = np.sqrt(np.sum((weighted - PIS) ** 2, axis=1))
        d_neg = np.sqrt(np.sum((weighted - NIS) ** 2, axis=1))

        closeness = d_neg / (d_pos + d_neg + 1e-12)
        best_idx = np.argmax(closeness)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'euclidean' or 'topsis'.")

    return solutions[best_idx], objectives[best_idx], best_idx
