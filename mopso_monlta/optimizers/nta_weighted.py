"""
Nonlinear Threshold Accepting with weighted-sum scalarization (baseline).

This converts the multi-objective problem into multiple single-objective
subproblems via random Dirichlet weight vectors. Kept for comparison
to demonstrate why true Pareto-based MONLTA is superior.

Limitation: weighted-sum cannot discover solutions on concave regions
of the Pareto front, and results depend on weight choice.
"""

import time
import numpy as np

from ..evaluation.pareto import dominates, crowding_distance


class NTA_WeightedSum:
    """
    NTA with weighted-sum scalarization.

    Decomposes the multi-objective problem into n_weight_vectors
    single-objective subproblems: minimize Σ w_i * f_i(x).

    Threshold schedule: T_k = T₀ * (1 - k/k_max)^α

    Args:
        n_iterations: Iterations per weight vector
        bounds: [(min, max)] per decision variable
        n_objectives: Number of objectives
        T0: Initial threshold amplitude
        alpha: Power-law cooling exponent
        n_weight_vectors: Number of random Dirichlet weight vectors
        step_size: Perturbation magnitude (fraction of range)
        archive_size: Maximum non-dominated solutions retained
    """

    def __init__(self, n_iterations=1000, bounds=None, n_objectives=3,
                 T0=1.0, alpha=2.0, n_weight_vectors=10,
                 step_size=0.1, archive_size=100):
        self.n_iterations = n_iterations
        self.bounds = bounds
        self.n_dim = len(bounds)
        self.n_objectives = n_objectives
        self.T0 = T0
        self.alpha = alpha
        self.n_weight_vectors = n_weight_vectors
        self.step_size = step_size
        self.archive_size = archive_size

        # Random Dirichlet weight vectors for decomposition
        self.weight_vectors = [np.random.dirichlet(np.ones(n_objectives))
                               for _ in range(n_weight_vectors)]

        self.archive_solutions = []
        self.archive_objectives = []
        self.history = []

    def _threshold(self, k, k_max):
        """Power-law cooling schedule."""
        return self.T0 * (1 - k / k_max) ** self.alpha

    def _weighted_sum(self, objectives, weights):
        """Scalarize objectives into a single value."""
        return sum(w * o for w, o in zip(weights, objectives))

    def _neighbor(self, x):
        """Generate neighbor via Gaussian perturbation."""
        x_new = x.copy()
        for d in range(self.n_dim):
            low, high = self.bounds[d]
            delta = (high - low) * self.step_size * np.random.randn()
            x_new[d] = np.clip(x_new[d] + delta, low, high)
        return x_new

    def _update_archive(self, solution, objectives):
        """Update archive with dominance checks and crowding pruning."""
        if objectives[0] >= 1e5:
            return

        is_dominated = False
        to_remove = []

        for i, arch_obj in enumerate(self.archive_objectives):
            if dominates(arch_obj, objectives):
                is_dominated = True
                break
            if dominates(objectives, arch_obj):
                to_remove.append(i)

        if not is_dominated:
            for i in sorted(to_remove, reverse=True):
                del self.archive_solutions[i]
                del self.archive_objectives[i]
            self.archive_solutions.append(solution.copy())
            self.archive_objectives.append(objectives.copy())

        if len(self.archive_solutions) > self.archive_size:
            distances = crowding_distance(self.archive_objectives)
            sorted_idx = np.argsort(distances)[::-1]
            self.archive_solutions = [self.archive_solutions[i]
                                      for i in sorted_idx[:self.archive_size]]
            self.archive_objectives = [self.archive_objectives[i]
                                       for i in sorted_idx[:self.archive_size]]

    def optimize(self, objective_func, verbose=True):
        """
        Run NTA with weighted-sum decomposition.

        Args:
            objective_func: callable(x) → [f1, f2, ...]
            verbose: print progress per weight vector

        Returns:
            (archive_solutions, archive_objectives)
        """
        start_time = time.time()
        total_evals = 0

        if verbose:
            print(f"NTA (Weighted-Sum) Started")
            print(f"  Weight vectors: {self.n_weight_vectors}, "
                  f"Iters/vector: {self.n_iterations}")
            print("-" * 60)

        for w_idx, weights in enumerate(self.weight_vectors):
            x_current = np.array([np.random.uniform(low, high)
                                  for low, high in self.bounds])
            obj_current = objective_func(x_current)
            f_current = self._weighted_sum(obj_current, weights)
            self._update_archive(x_current, obj_current)
            total_evals += 1

            for k in range(self.n_iterations):
                x_new = self._neighbor(x_current)
                obj_new = objective_func(x_new)
                f_new = self._weighted_sum(obj_new, weights)
                total_evals += 1

                T = self._threshold(k, self.n_iterations)
                if f_new - f_current < T:
                    x_current, obj_current, f_current = x_new, obj_new, f_new
                    self._update_archive(x_current, obj_current)

            self.history.append({
                'weight_idx': w_idx,
                'archive_size': len(self.archive_solutions),
            })

            if verbose:
                print(f"  Weight {w_idx+1:2d}/{self.n_weight_vectors}: "
                      f"Archive = {len(self.archive_solutions):3d}")

        elapsed = time.time() - start_time
        if verbose:
            print("-" * 60)
            print(f"Complete in {elapsed:.2f}s | Evals: {total_evals} "
                  f"| Pareto size: {len(self.archive_solutions)}")

        return self.archive_solutions, self.archive_objectives
