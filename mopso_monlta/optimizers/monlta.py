"""
Multi-Objective Nonlinear Threshold Accepting (MONLTA).

Reference:
    Nahas, Abouheaf, Darghouth, Sharaf, "A multi-objective AVR-LFC
    optimization scheme for multi-area power systems,"
    Electric Power Systems Research, 200, 107467 (2021).

Key innovations over standard Threshold Accepting:
    - Nonlinear accepting function: H(ζ) = 1/√(1 + (ζ/ζ₀)²)
    - Four dominance-based acceptance scenarios
    - Amount-of-domination principle for nuanced acceptance
    - Variable-size Pareto archive (no weight factors needed)
    - One-variable-at-a-time perturbation (focused search)
"""

import time
import numpy as np

from ..evaluation.pareto import dominates, crowding_distance


class MONLTA:
    """
    Multi-Objective Nonlinear Threshold Accepting (Nahas et al., 2021).

    A true Pareto-based local search using a low-pass-filter accepting
    function with four acceptance scenarios based on dominance relations.

    Args:
        n_iterations: Iterations per episode (NI in paper)
        bounds: [(min, max)] for each decision variable
        n_objectives: Number of objectives
        zeta_0_inv: 1/ζ₀ = 0.0075 (paper's value)
        zeta_start: Initial frequency parameter (paper: 40.0)
        archive_init_size: L — initial random archive seeds (paper: 5)
        archive_max_size: Hard cap on archive storage
        n_episodes: Independent search restarts
        perturbation_mode: 'single' (paper's one-variable-at-a-time) or 'gaussian'
    """

    def __init__(self, n_iterations=1000, bounds=None, n_objectives=3,
                 zeta_0_inv=0.0075, zeta_start=40.0, archive_init_size=5,
                 archive_max_size=100, n_episodes=10, perturbation_mode='single'):
        self.n_iterations = n_iterations
        self.bounds = bounds
        self.n_dim = len(bounds)
        self.n_objectives = n_objectives
        self.zeta_0_inv = zeta_0_inv
        self.zeta_0 = 1.0 / zeta_0_inv
        self.zeta_start = zeta_start
        self.delta_zeta = zeta_start / n_iterations
        self.archive_init_size = archive_init_size
        self.archive_max_size = archive_max_size
        self.n_episodes = n_episodes
        self.perturbation_mode = perturbation_mode

        self.archive_solutions = []
        self.archive_objectives = []
        self.history = []
        self.episode_history = []

    def H(self, zeta):
        """
        Nonlinear accepting function (low-pass filter form).

        H(ζ) = 1 / √(1 + (ζ/ζ₀)²)

        Transitions from ~1 (accept easily) to ~0 (only improvements).
        """
        return 1.0 / np.sqrt(1.0 + (zeta / self.zeta_0) ** 2)

    def generate_neighbor(self, x):
        """Generate neighbor by perturbing one variable (paper's approach)."""
        x_new = x.copy()

        if self.perturbation_mode == 'single':
            d = np.random.randint(self.n_dim)
            low, high = self.bounds[d]
            x_new[d] = np.random.uniform(low, high)
        else:
            for d in range(self.n_dim):
                low, high = self.bounds[d]
                delta = (high - low) * 0.1 * np.random.randn()
                x_new[d] = np.clip(x_new[d] + delta, low, high)

        return x_new

    def _find_dominating_archive_members(self, obj):
        """Find archive members that dominate the given objective vector."""
        return [i for i, arch_obj in enumerate(self.archive_objectives)
                if dominates(arch_obj, obj)]

    def _amount_of_domination_avg(self, obj_neighbor, dominating_indices):
        """Compute amount-of-domination average: O_i^Av(X) = [O_i(X') - Σ O_i(X_j)] / (k+1)."""
        k = len(dominating_indices)
        if k == 0:
            return obj_neighbor

        avg = []
        for i in range(self.n_objectives):
            sum_dom = sum(self.archive_objectives[j][i] for j in dominating_indices)
            avg.append((obj_neighbor[i] - sum_dom) / (k + 1))
        return avg

    def update_archive(self, solution, objectives):
        """Update variable-size archive with dominance rules."""
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

        if len(self.archive_solutions) > self.archive_max_size:
            distances = crowding_distance(self.archive_objectives)
            sorted_idx = np.argsort(distances)[::-1]
            self.archive_solutions = [self.archive_solutions[i]
                                      for i in sorted_idx[:self.archive_max_size]]
            self.archive_objectives = [self.archive_objectives[i]
                                       for i in sorted_idx[:self.archive_max_size]]

    def acceptance_check(self, obj_current, obj_neighbor, zeta):
        """
        Implement the four acceptance scenarios from the paper.

        Scenario 1: Neighbor dominates current, not dominated by archive → Accept
        Scenario 2: Neighbor dominates current, dominated by k archive members → Accept
        Scenario 3: Neighbor ≠ dominate current, dominated by k members → Conditional
        Scenario 4: Neighbor ≠ dominate current, not dominated by archive → Conditional

        Returns:
            (accept: bool, scenario: int)
        """
        neighbor_dom_current = dominates(obj_neighbor, obj_current)
        dom_indices = self._find_dominating_archive_members(obj_neighbor)
        k = len(dom_indices)
        H_zeta = self.H(zeta)

        # Scenario 1
        if neighbor_dom_current and k == 0:
            return True, 1

        # Scenario 2
        if neighbor_dom_current and k > 0:
            return True, 2

        # Scenario 3
        if not neighbor_dom_current and k > 0:
            o_avg = self._amount_of_domination_avg(obj_neighbor, dom_indices)
            for i in range(self.n_objectives):
                if obj_current[i] > 0:
                    if abs(o_avg[i]) / abs(obj_current[i]) > H_zeta:
                        return False, 3
                elif o_avg[i] > 0:
                    return False, 3
            return True, 3

        # Scenario 4
        for i in range(self.n_objectives):
            if obj_current[i] > 0:
                if obj_neighbor[i] / obj_current[i] > (1.0 + H_zeta):
                    return False, 4
            elif obj_neighbor[i] > 0:
                return False, 4
        return True, 4

    def optimize(self, objective_func, verbose=True):
        """
        Run MONLTA optimization over multiple episodes.

        Each episode performs n_iterations of focused perturbation with
        the threshold progressively tightening (ζ decreasing).

        Args:
            objective_func: callable(x) → [f1, f2, ...]
            verbose: print progress per episode

        Returns:
            (archive_solutions, archive_objectives)
        """
        start_time = time.time()
        total_evals = 0
        scenario_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        accepted_counts = {1: 0, 2: 0, 3: 0, 4: 0}

        if verbose:
            print(f"MONLTA Optimization Started (Nahas et al. 2021)")
            print(f"  Episodes: {self.n_episodes}, Iterations/episode: {self.n_iterations}")
            print(f"  ζ₀ = {self.zeta_0:.2f}, ζ_start = {self.zeta_start}")
            print(f"  H(ζ_start) = {self.H(self.zeta_start):.4f}")
            print("-" * 65)

        for episode in range(self.n_episodes):
            x_current = np.array([np.random.uniform(low, high)
                                  for low, high in self.bounds])
            obj_current = objective_func(x_current)
            total_evals += 1
            self.update_archive(x_current, obj_current)

            # Seed archive on first episode
            if episode == 0 and len(self.archive_solutions) < self.archive_init_size:
                for _ in range(self.archive_init_size - len(self.archive_solutions)):
                    x_init = np.array([np.random.uniform(low, high)
                                       for low, high in self.bounds])
                    obj_init = objective_func(x_init)
                    total_evals += 1
                    self.update_archive(x_init, obj_init)

            zeta = self.zeta_start
            episode_accepts = 0

            for k in range(self.n_iterations):
                x_neighbor = self.generate_neighbor(x_current)
                obj_neighbor = objective_func(x_neighbor)
                total_evals += 1

                accept, scenario = self.acceptance_check(
                    obj_current, obj_neighbor, zeta)

                scenario_counts[scenario] += 1

                if accept:
                    x_current = x_neighbor
                    obj_current = obj_neighbor
                    episode_accepts += 1
                    accepted_counts[scenario] += 1
                    self.update_archive(x_current, obj_current)

                zeta = max(zeta - self.delta_zeta, self.delta_zeta)

            self.episode_history.append({
                'episode': episode,
                'archive_size': len(self.archive_solutions),
                'accepts': episode_accepts,
                'accept_rate': episode_accepts / self.n_iterations,
                'best_obj': [min(obj[i] for obj in self.archive_objectives)
                             for i in range(self.n_objectives)]
                if self.archive_objectives else None,
            })

            if verbose:
                best = self.episode_history[-1]['best_obj']
                best_str = ", ".join(f"{b:.4f}" for b in best) if best else "N/A"
                print(f"  Episode {episode+1:2d}/{self.n_episodes}: "
                      f"Archive = {len(self.archive_solutions):3d}, "
                      f"Accept = {episode_accepts/self.n_iterations:.1%}, "
                      f"Best = [{best_str}]")

        elapsed = time.time() - start_time
        if verbose:
            print("-" * 65)
            print(f"Complete in {elapsed:.2f}s | Evals: {total_evals} "
                  f"| Pareto size: {len(self.archive_solutions)}")
            print(f"\nScenario statistics (triggered / accepted):")
            for s in range(1, 5):
                print(f"  Scenario {s}: {scenario_counts[s]:5d} / {accepted_counts[s]:5d}")

        return self.archive_solutions, self.archive_objectives
