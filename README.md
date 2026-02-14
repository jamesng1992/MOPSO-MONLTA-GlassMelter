# Multi-Objective Optimization for Glass Melter Control: MOPSO, NTA & MONLTA

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A51.21-orange.svg)](https://numpy.org/)

## Overview

This repository implements and compares **three multi-objective heuristic optimization algorithms** for PID controller tuning in glass melting furnace level control systems. The central contribution is the adaptation of the **MONLTA** (Multi-Objective Nonlinear Threshold Accepting) algorithm from power systems to glass manufacturing applications.

The framework is then extended to **five advanced glass melter control problems** including Neural ODE hyperparameter optimization, observer gain tuning, MPC weight selection, fractional-order PID design, and simultaneous multi-zone tuning.

### Algorithms Implemented

| # | Method | Family | Multi-Objective Strategy |
|---|--------|--------|--------------------------|
| 1 | **MOPSO** | Population-based (swarm) | External Pareto archive + crowding distance leader selection |
| 2 | **NTA** (Weighted-Sum) | Local search (SA-family) | Weighted-sum scalarization with multiple weight vectors |
| 3 | **MONLTA** (Paper) | Local search (SA-family) | True Pareto-based with 4 acceptance scenarios |

## Problem Statement

### Physical System

A **glass melting furnace** continuously melts raw batch material and delivers molten glass to forming operations. The **glass level** $h(t)$ must be tightly controlled:
- **Too low** → air entrainment, poor glass quality
- **Too high** → overflow risk, safety hazard

### Simplified Dynamics

The level is modeled by a second-order ODE:

$$\frac{dh}{dt} = v, \quad \frac{dv}{dt} = -\frac{1}{\tau_m}v + \frac{K_m}{\tau_m}(u - q_p)$$

where $u(t)$ is the batch charging rate (control input) and $q_p(t)$ is the production pull (disturbance).

### Optimization Objectives

The decision variables are the PID gains $(K_p, K_i, K_d)$, and the objectives (all minimized) are:

$$\min_{K_p, K_i, K_d} \mathbf{f}(\mathbf{x}) = \begin{bmatrix} \text{IAE (Integral Absolute Error)} \\ \text{Overshoot (\%)} \\ \text{Settling Time (h)} \end{bmatrix}$$

Additional performance metrics computed: ISE, ITSE, ITAE, rise time, peak time, 2%- and 5%-band settling times.

## Repository Structure

```
├── MultiObjective_Optimization_PSO_NTA.ipynb      # Core algorithms & PID tuning
├── MOPSO_MONLTA_Applications_GlassMelter.ipynb     # Five advanced applications
├── README.md
├── LICENSE
└── .gitignore
```

## Notebooks

### 1. `MultiObjective_Optimization_PSO_NTA.ipynb` — Core Framework

This notebook develops the complete multi-objective optimization framework:

- **Glass melter dynamics** — Second-order level model with PID controller and anti-windup
- **MOPSO** — Particle Swarm Optimization with external Pareto archive, crowding distance-based leader selection, and polynomial mutation
- **NTA** — Weighted-sum scalarization approach with nonlinear threshold accepting
- **MONLTA** — True Pareto-based optimization with the 4-scenario acceptance criterion from Nahas et al. (2021)
- **Visualization** — 2D/3D Pareto fronts, convergence curves, closed-loop step responses
- **TOPSIS** — Technique for Order of Preference by Similarity to Ideal Solution for systematic solution selection

#### Key MONLTA Features (from the paper)

- **Nonlinear Accepting Function**: $H(\zeta) = 1/\sqrt{1+(\zeta/\zeta_0)^2}$ — a low-pass-filter form providing controlled exploration-to-exploitation transition
- **Four Acceptance Scenarios**: Dominance-based acceptance with an amount-of-domination principle
- **Variable-Size Archive**: Non-dominated solutions stored and updated using dominance rules
- **Focused Perturbation**: One decision variable perturbed at a time for targeted neighborhood search

### 2. `MOPSO_MONLTA_Applications_GlassMelter.ipynb` — Advanced Applications

This notebook applies the MOPSO/MONLTA framework to five glass melter control challenges using a 7-state ODE model:

| # | Application | Decision Variables | Objectives |
|---|-------------|-------------------|------------|
| 1 | **Neural ODE Hyperparameter Optimization** | hidden_dim, n_layers, lr, correction_scale | Validation RMSE, Parameter count, Training time |
| 2 | **Observer Gain Tuning** | $L_1, \ldots, L_7$ (Luenberger gains) | Convergence speed, Noise sensitivity, Robustness |
| 3 | **MPC Weight Selection** | $Q_{1..7}$, $R$, $N_p$ | Tracking error, Control effort, Constraint violations |
| 4 | **FOPID Controller Design** | $K_p, K_i, K_d, \lambda, \mu$ | ITSE, Overshoot, Settling time |
| 5 | **Simultaneous Multi-Zone Tuning** | PID gains for $n$ zones | Per-zone IAE, Cross-coupling, Total energy |

## Requirements

- Python ≥ 3.8
- NumPy
- Matplotlib
- SciPy

Install dependencies:

```bash
pip install numpy matplotlib scipy
```

## Quick Start

```bash
git clone https://github.com/jamesng1992/MOPSO-MONLTA-GlassMelter.git
cd MOPSO-MONLTA-GlassMelter
pip install numpy matplotlib scipy
jupyter notebook MultiObjective_Optimization_PSO_NTA.ipynb
```

## Key References

1. Nahas, Abouheaf, Darghouth, Sharaf, "A multi-objective AVR-LFC optimization scheme for multi-area power systems," *Electric Power Systems Research*, 200, 107467 (2021).
2. Coello et al., "Handling Multiple Objectives with Particle Swarm Optimization," *IEEE Transactions on Evolutionary Computation*, 2004.
3. Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II," *IEEE Transactions on Evolutionary Computation*, 2002.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

James Ng — [GitHub](https://github.com/jamesng1992)
