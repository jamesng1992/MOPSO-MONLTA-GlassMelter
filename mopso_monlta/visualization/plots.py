"""
Reusable plotting functions for multi-objective optimization results.

Functions for visualizing Pareto fronts (2D/3D), controller step responses,
convergence analysis, and MONLTA-specific diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .style import COLORS


def plot_pareto_2d(results_dict, obj_names=('IAE', 'Overshoot (%)', 'Settling Time (h)'),
                   save_path=None):
    """
    Plot 2D Pareto front projections for multiple methods.

    Creates three subplots showing every pairwise combination of objectives.

    Args:
        results_dict: {method_name: objectives_list}
            e.g. {'MOPSO': [[f1,f2,f3], ...], 'MONLTA': [...]}
        obj_names: tuple of objective axis labels
        save_path: optional path to save figure
    """
    color_cycle = [COLORS['blue'], COLORS['orange'], COLORS['green'],
                   COLORS['purple'], COLORS['cyan']]
    markers = ['o', 's', 'D', '^', 'v']
    pairs = [(0, 1), (0, 2), (1, 2)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for panel, (i, j) in enumerate(pairs):
        ax = axes[panel]
        for k, (name, objs) in enumerate(results_dict.items()):
            x = [o[i] for o in objs]
            y = [o[j] for o in objs]
            ax.scatter(x, y, c=color_cycle[k % len(color_cycle)],
                       s=50, alpha=0.7, label=name,
                       edgecolors='black', linewidth=0.5,
                       marker=markers[k % len(markers)])
        ax.set_xlabel(obj_names[i])
        ax.set_ylabel(obj_names[j])
        ax.set_title(f'{obj_names[i]} vs. {obj_names[j]}')
        ax.legend(fontsize=8)

    plt.suptitle('Pareto Front Comparison', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_pareto_3d(results_dict, obj_names=('IAE', 'Overshoot (%)', 'Settling Time (h)'),
                   save_path=None):
    """
    Plot 3D Pareto surfaces for each method in separate subplots.

    Args:
        results_dict: {method_name: objectives_list}
        obj_names: tuple of 3 objective axis labels
        save_path: optional path to save figure
    """
    color_cycle = [COLORS['blue'], COLORS['orange'], COLORS['green'],
                   COLORS['purple'], COLORS['cyan']]
    n_methods = len(results_dict)
    fig = plt.figure(figsize=(6 * n_methods, 5))

    for k, (name, objs) in enumerate(results_dict.items()):
        ax = fig.add_subplot(1, n_methods, k + 1, projection='3d')
        x = [o[0] for o in objs]
        y = [o[1] for o in objs]
        z = [o[2] for o in objs]
        ax.scatter(x, y, z, c=color_cycle[k % len(color_cycle)],
                   s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.set_xlabel(obj_names[0])
        ax.set_ylabel(obj_names[1])
        ax.set_zlabel(obj_names[2])
        ax.set_title(name)

    plt.suptitle('3D Pareto Fronts', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_controller_comparison(sim_results, save_path=None):
    """
    Plot closed-loop level response and control effort for multiple controllers.

    Args:
        sim_results: list of (name, t, h, u, color, linestyle) tuples
        save_path: optional path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1 = axes[0]
    ax1.axhline(y=0.9, color='gray', linestyle='--', linewidth=1, label='Setpoint')
    ax1.axvline(x=10, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    for name, t, h, u, color, ls in sim_results:
        ax1.plot(t, h, color=color, linewidth=2, linestyle=ls, label=name)

    ax1.set_ylabel('Level $h$ (m)')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_ylim([0.84, 0.96])
    ax1.set_title('Closed-Loop Level Response Comparison')

    ax2 = axes[1]
    ax2.axvline(x=10, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    for name, t, h, u, color, ls in sim_results:
        ax2.plot(t, u, color=color, linewidth=2, linestyle=ls, label=name)

    ax2.set_xlabel('Time (h)')
    ax2.set_ylabel('Control $u$ (t/h)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_title('Control Effort Comparison')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_convergence(mopso, nta, monlta, save_path=None):
    """
    Plot convergence analysis for all three optimizers.

    Args:
        mopso: MOPSO instance (with .history)
        nta: NTA_WeightedSum instance (with .history)
        monlta: MONLTA instance (with .episode_history)
        save_path: optional save path
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # MOPSO
    ax1 = axes[0]
    iters = [h['iteration'] for h in mopso.history]
    arch = [h['archive_size'] for h in mopso.history]
    best = [h['best_f1'] for h in mopso.history]

    ax1_twin = ax1.twinx()
    ax1.plot(iters, arch, color=COLORS['blue'], linewidth=2, label='Archive Size')
    ax1_twin.plot(iters, best, color=COLORS['cyan'], linewidth=2,
                  linestyle='--', label='Best f1')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Archive Size', color=COLORS['blue'])
    ax1_twin.set_ylabel('Best f1', color=COLORS['cyan'])
    ax1.set_title('MOPSO Convergence')

    # NTA
    ax2 = axes[1]
    w_idx = [h['weight_idx'] for h in nta.history]
    arch_nta = [h['archive_size'] for h in nta.history]
    ax2.bar(w_idx, arch_nta, color=COLORS['orange'], alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Weight Vector Index')
    ax2.set_ylabel('Archive Size')
    ax2.set_title('NTA (Weighted) Archive Growth')

    # MONLTA
    ax3 = axes[2]
    episodes = [h['episode'] for h in monlta.episode_history]
    arch_m = [h['archive_size'] for h in monlta.episode_history]
    rates = [h['accept_rate'] for h in monlta.episode_history]

    ax3_twin = ax3.twinx()
    ax3.bar(episodes, arch_m, color=COLORS['green'], alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax3_twin.plot(episodes, rates, color=COLORS['purple'], linewidth=2,
                  marker='o', markersize=4)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Archive Size', color=COLORS['green'])
    ax3_twin.set_ylabel('Accept Rate', color=COLORS['purple'])
    ax3.set_title('MONLTA Convergence')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_monlta_analysis(monlta, save_path=None):
    """
    Plot MONLTA accepting function H(ζ) and per-episode statistics.

    Args:
        monlta: MONLTA instance
        save_path: optional save path
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # H(ζ) curve
    ax1 = axes[0]
    zeta_range = np.linspace(0.01, 60, 500)
    H_values = 1.0 / np.sqrt(1.0 + (zeta_range / monlta.zeta_0) ** 2)

    ax1.plot(zeta_range, H_values, color=COLORS['green'], linewidth=2.5)
    ax1.axhline(y=monlta.H(monlta.zeta_start), color='gray', linestyle='--',
                alpha=0.5, label=f'H(ζ_start={monlta.zeta_start}) = {monlta.H(monlta.zeta_start):.4f}')
    ax1.axhline(y=monlta.H(monlta.delta_zeta), color='gray', linestyle=':',
                alpha=0.5, label=f'H(Δζ={monlta.delta_zeta:.4f}) = {monlta.H(monlta.delta_zeta):.4f}')
    ax1.set_xlabel(r'$\zeta$ (frequency parameter)')
    ax1.set_ylabel(r'$H(\zeta)$ — Acceptance Threshold')
    ax1.set_title(r'$H(\zeta) = 1/\sqrt{1+(\zeta/\zeta_0)^2}$')
    ax1.legend(fontsize=8)
    ax1.annotate(f'ζ₀ = {monlta.zeta_0:.1f}', xy=(monlta.zeta_0, 0.707),
                 xytext=(monlta.zeta_0 + 10, 0.75),
                 arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)

    # Episode statistics
    ax2 = axes[1]
    episodes = [h['episode'] + 1 for h in monlta.episode_history]
    rates = [h['accept_rate'] * 100 for h in monlta.episode_history]
    arch = [h['archive_size'] for h in monlta.episode_history]

    bars = ax2.bar(episodes, rates, color=COLORS['green'], alpha=0.6,
                   edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Accept Rate (%)', color=COLORS['green'])

    ax2_twin = ax2.twinx()
    ax2_twin.plot(episodes, arch, color=COLORS['purple'], linewidth=2.5,
                  marker='D', markersize=5)
    ax2_twin.set_ylabel('Archive Size', color=COLORS['purple'])
    ax2.set_title('MONLTA Episode Statistics')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
