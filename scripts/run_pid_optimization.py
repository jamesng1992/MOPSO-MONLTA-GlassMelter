#!/usr/bin/env python
"""
Run PID optimization for the glass melter using MOPSO, NTA, and MONLTA.

Example usage:
    python scripts/run_pid_optimization.py
    python scripts/run_pid_optimization.py --n-particles 40 --n-iterations 80
"""

import argparse
import numpy as np
import sys
import os

# Ensure package is importable when running as script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mopso_monlta.config import DEFAULT_PID_BOUNDS, SIM_DEFAULTS
from mopso_monlta.optimizers import MOPSO, NTA_WeightedSum, MONLTA
from mopso_monlta.evaluation import (
    evaluate_pid_objectives,
    get_pareto_front,
    select_best_compromise,
)
from mopso_monlta.models import simulate_glass_melter, PIDController
from mopso_monlta.visualization import (
    setup_publication_style,
    plot_pareto_2d,
    plot_controller_comparison,
    plot_convergence,
)


def run_optimization(n_particles=30, n_iterations=60, n_weights=12, n_episodes=5):
    """Run the three-method comparison and display results."""
    setup_publication_style()
    np.random.seed(42)

    bounds = DEFAULT_PID_BOUNDS
    n_obj = 3

    print("=" * 60)
    print("  Multi-Objective PID Tuning for Glass Melter")
    print("=" * 60)

    # --- MOPSO ---
    print(f"\n[1/3] Running MOPSO ({n_particles} particles, {n_iterations} iterations)...")
    mopso = MOPSO(
        objective_func=evaluate_pid_objectives,
        bounds=bounds,
        n_particles=n_particles,
        n_objectives=n_obj,
        archive_size=100,
    )
    mopso.optimize(n_iterations=n_iterations)
    pareto_mopso = mopso.archive_objectives
    gains_mopso = mopso.archive_positions
    print(f"    Archive size: {len(pareto_mopso)}")

    # --- NTA (Weighted Sum) ---
    print(f"\n[2/3] Running NTA Weighted Sum ({n_weights} weight vectors)...")
    nta = NTA_WeightedSum(
        objective_func=evaluate_pid_objectives,
        bounds=bounds,
        n_objectives=n_obj,
    )
    nta.optimize(n_weight_vectors=n_weights, n_iterations=200)
    pareto_nta = nta.archive_objectives
    gains_nta = nta.archive_positions
    print(f"    Archive size: {len(pareto_nta)}")

    # --- MONLTA ---
    print(f"\n[3/3] Running MONLTA ({n_episodes} episodes)...")
    monlta = MONLTA(
        objective_func=evaluate_pid_objectives,
        bounds=bounds,
        n_objectives=n_obj,
    )
    monlta.optimize(n_episodes=n_episodes, steps_per_episode=60)
    pareto_monlta = monlta.archive_objectives
    gains_monlta = monlta.archive_positions
    print(f"    Archive size: {len(pareto_monlta)}")

    # --- Best compromise via TOPSIS ---
    print("\n" + "-" * 60)
    print("  Selecting Best Compromise Solutions (TOPSIS)")
    print("-" * 60)

    methods = {
        "MOPSO": (gains_mopso, pareto_mopso),
        "NTA": (gains_nta, pareto_nta),
        "MONLTA": (gains_monlta, pareto_monlta),
    }

    from mopso_monlta.visualization.style import COLORS
    sim_results = []
    color_list = [COLORS['blue'], COLORS['orange'], COLORS['green']]
    ls_list = ['-', '--', '-.']

    for idx, (name, (gains, objs)) in enumerate(methods.items()):
        best_gains, best_obj = select_best_compromise(gains, objs)
        print(f"\n  {name}:")
        print(f"    Kp={best_gains[0]:.4f}, Ki={best_gains[1]:.4f}, Kd={best_gains[2]:.4f}")
        print(f"    IAE={best_obj[0]:.4f}, OS={best_obj[1]:.2f}%, Ts={best_obj[2]:.4f} h")

        # Simulate for plotting
        pid = PIDController(
            Kp=best_gains[0], Ki=best_gains[1], Kd=best_gains[2],
            dt=SIM_DEFAULTS['dt'], u_min=0.0, u_max=SIM_DEFAULTS['u_max'],
        )
        t, h, u = simulate_glass_melter(
            pid, t_end=SIM_DEFAULTS['t_end'], dt=SIM_DEFAULTS['dt'],
            setpoint=SIM_DEFAULTS['setpoint'],
            disturbance_time=SIM_DEFAULTS['disturbance_time'],
            disturbance_mag=SIM_DEFAULTS['disturbance_mag'],
        )
        sim_results.append((name, t, h, u, color_list[idx], ls_list[idx]))

    # --- Plots ---
    print("\n\nGenerating plots...")
    results_dict = {
        "MOPSO": pareto_mopso,
        "NTA": pareto_nta,
        "MONLTA": pareto_monlta,
    }
    plot_pareto_2d(results_dict, save_path="docs/figures/pareto_2d.png")
    plot_controller_comparison(sim_results, save_path="docs/figures/controller_comparison.png")
    plot_convergence(mopso, nta, monlta, save_path="docs/figures/convergence.png")

    print("\nDone! Figures saved to docs/figures/")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Objective PID Tuning for Glass Melter"
    )
    parser.add_argument("--n-particles", type=int, default=30)
    parser.add_argument("--n-iterations", type=int, default=60)
    parser.add_argument("--n-weights", type=int, default=12)
    parser.add_argument("--n-episodes", type=int, default=5)
    args = parser.parse_args()

    run_optimization(
        n_particles=args.n_particles,
        n_iterations=args.n_iterations,
        n_weights=args.n_weights,
        n_episodes=args.n_episodes,
    )


if __name__ == "__main__":
    main()
