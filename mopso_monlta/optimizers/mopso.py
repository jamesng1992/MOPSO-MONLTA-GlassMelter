"""
Multi-Objective Particle Swarm Optimization (MOPSO).

Reference:
    Coello et al., "Handling Multiple Objectives with Particle Swarm
    Optimization," IEEE Trans. Evolutionary Computation, 2004.

Key features:
    - External Pareto archive with crowding-distance pruning
    - Leader selection via crowding-distance-weighted roulette wheel
    - Polynomial mutation for diversity preservation
"""

import time
import numpy as np
from copy import deepcopy

from ..evaluation.pareto import dominates, crowding_distance


class MOPSO:
    """
    Multi-Objective Particle Swarm Optimization.

    The swarm evolves over n_iterations, building a Pareto archive
    of non-dominated solutions. The archive IS the approximate
    Pareto front at convergence.

    Args:
        n_particles: Swarm size
        n_iterations: Number of PSO generations
        bounds: [(min, max)] for each decision variable
        n_objectives: Number of objectives
        archive_size: Maximum Pareto archive capacity
        w: Inertia weight (< 1 promotes convergence)
        c1: Cognitive coefficient (personal best attraction)
        c2: Social coefficient (archive leader attraction)
        mutation_prob: Per-dimension polynomial mutation probability
    """

    def __init__(self, n_particles=50, n_iterations=100,
                 bounds=None, n_objectives=3, archive_size=100,
                 w=0.4, c1=2.0, c2=2.0, mutation_prob=0.1):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.bounds = bounds
        self.n_dim = len(bounds)
        self.n_objectives = n_objectives
        self.archive_size = archive_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.mutation_prob = mutation_prob

        self.archive_solutions = []
        self.archive_objectives = []
        self.history = []

    def initialize(self):
        """Initialize particle positions and velocities uniformly within bounds."""
        self.positions = np.zeros((self.n_particles, self.n_dim))
        self.velocities = np.zeros((self.n_particles, self.n_dim))

        for d in range(self.n_dim):
            low, high = self.bounds[d]
            self.positions[:, d] = np.random.uniform(low, high, self.n_particles)
            vel_range = (high - low) * 0.1
            self.velocities[:, d] = np.random.uniform(-vel_range, vel_range,
                                                       self.n_particles)

        self.pbest_positions = self.positions.copy()
        self.pbest_objectives = [None] * self.n_particles

    def update_archive(self, solutions, objectives):
        """Update external Pareto archive with dominance checks and crowding pruning."""
        for sol, obj in zip(solutions, objectives):
            if obj[0] >= 1e5:
                continue

            is_dominated = False
            to_remove = []

            for i, arch_obj in enumerate(self.archive_objectives):
                if dominates(arch_obj, obj):
                    is_dominated = True
                    break
                if dominates(obj, arch_obj):
                    to_remove.append(i)

            if not is_dominated:
                for i in sorted(to_remove, reverse=True):
                    del self.archive_solutions[i]
                    del self.archive_objectives[i]
                self.archive_solutions.append(sol.copy())
                self.archive_objectives.append(obj.copy())

        # Prune via crowding distance (keep most diverse solutions)
        if len(self.archive_solutions) > self.archive_size:
            distances = crowding_distance(self.archive_objectives)
            sorted_idx = np.argsort(distances)[::-1]
            self.archive_solutions = [self.archive_solutions[i]
                                      for i in sorted_idx[:self.archive_size]]
            self.archive_objectives = [self.archive_objectives[i]
                                       for i in sorted_idx[:self.archive_size]]

    def select_leader(self):
        """Select archive leader via crowding-distance-weighted roulette wheel."""
        if len(self.archive_solutions) == 0:
            return None

        distances = crowding_distance(self.archive_objectives)
        total = sum(d for d in distances if d != float('inf'))
        if total == 0:
            return self.archive_solutions[np.random.randint(len(self.archive_solutions))]

        probs = [d / total if d != float('inf') else 1.0 for d in distances]
        probs = np.array(probs) / sum(probs)
        idx = np.random.choice(len(self.archive_solutions), p=probs)
        return self.archive_solutions[idx]

    def mutate(self, position):
        """Apply Gaussian mutation for diversity preservation."""
        mutated = position.copy()
        for d in range(self.n_dim):
            if np.random.random() < self.mutation_prob:
                low, high = self.bounds[d]
                delta = (high - low) * 0.1 * np.random.randn()
                mutated[d] = np.clip(mutated[d] + delta, low, high)
        return mutated

    def optimize(self, objective_func, verbose=True):
        """
        Run the MOPSO optimization loop.

        Args:
            objective_func: callable(x) → [f1, f2, ...]
            verbose: print progress every 10 iterations

        Returns:
            (archive_solutions, archive_objectives)
        """
        start_time = time.time()
        self.initialize()

        # Evaluate initial swarm
        objectives = [objective_func(self.positions[i])
                      for i in range(self.n_particles)]
        self.pbest_objectives = deepcopy(objectives)
        self.update_archive(self.positions, objectives)

        if verbose:
            print(f"MOPSO Optimization Started")
            print(f"  Particles: {self.n_particles}, Iterations: {self.n_iterations}")
            print(f"  Budget: {self.n_particles * self.n_iterations} evaluations")
            print("-" * 60)

        # Main PSO loop
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                gbest = self.select_leader()
                if gbest is None:
                    gbest = self.positions[i]

                # Velocity update
                r1 = np.random.random(self.n_dim)
                r2 = np.random.random(self.n_dim)
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social = self.c2 * r2 * (gbest - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social

                # Position update
                self.positions[i] += self.velocities[i]

                # Enforce bounds
                for d in range(self.n_dim):
                    low, high = self.bounds[d]
                    self.positions[i, d] = np.clip(self.positions[i, d], low, high)

                # Mutation
                self.positions[i] = self.mutate(self.positions[i])

                # Evaluate
                new_obj = objective_func(self.positions[i])
                objectives[i] = new_obj

                # Update personal best
                if dominates(new_obj, self.pbest_objectives[i]):
                    self.pbest_positions[i] = self.positions[i].copy()
                    self.pbest_objectives[i] = new_obj.copy()
                elif not dominates(self.pbest_objectives[i], new_obj):
                    if np.random.random() < 0.5:
                        self.pbest_positions[i] = self.positions[i].copy()
                        self.pbest_objectives[i] = new_obj.copy()

            self.update_archive(self.positions, objectives)

            self.history.append({
                'iteration': iteration,
                'archive_size': len(self.archive_solutions),
                'best_f1': min(obj[0] for obj in self.archive_objectives)
                if self.archive_objectives else float('inf'),
            })

            if verbose and (iteration + 1) % 10 == 0:
                print(f"  Iter {iteration+1:3d}: Archive = {len(self.archive_solutions):3d}, "
                      f"Best f1 = {self.history[-1]['best_f1']:.4f}")

        elapsed = time.time() - start_time
        if verbose:
            print("-" * 60)
            print(f"Complete in {elapsed:.2f}s | Pareto size: {len(self.archive_solutions)}")

        return self.archive_solutions, self.archive_objectives
