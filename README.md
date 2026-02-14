# Multi-Objective Optimization for Glass Melter Control: MOPSO, NTA & MONLTA

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A51.24-orange.svg)](https://numpy.org/)

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

$$\min_{K_p, K_i, K_d} \mathbf{f}(\mathbf{x}) = [\text{IAE (Integral Absolute Error)}, \\ \text{Overshoot (\%)}, \\ \text{Settling Time (h)}]$$

Additional performance metrics computed: ISE, ITSE, ITAE, rise time, peak time, 2%- and 5%-band settling times.

## Repository Structure

```
MOPSO-MONLTA-GlassMelter/
├── mopso_monlta/                   # Installable Python package
│   ├── __init__.py
│   ├── config.py                   # Physical parameters & defaults
│   ├── models/
│   │   ├── __init__.py
│   │   ├── glass_melter.py         # 2-state & 7-state ODE models
│   │   └── pid.py                  # PID controller with anti-windup
│   ├── optimizers/
│   │   ├── __init__.py
│   │   ├── mopso.py                # MOPSO (Coello et al. 2004)
│   │   ├── monlta.py               # MONLTA (Nahas et al. 2021)
│   │   └── nta_weighted.py         # NTA with weighted-sum scalarization
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── objectives.py           # Closed-loop simulation & metrics
│   │   ├── pareto.py               # Dominance, crowding distance
│   │   └── decision.py             # TOPSIS & compromise selection
│   └── visualization/
│       ├── __init__.py
│       ├── style.py                # Publication-quality plot settings
│       └── plots.py                # Pareto, convergence & response plots
├── notebooks/
│   ├── 01_pid_tuning.ipynb         # Core PID tuning comparison
│   └── 02_advanced_applications.ipynb  # 5 extended applications
├── scripts/
│   └── run_pid_optimization.py     # CLI script for quick runs
├── docs/figures/                   # Generated figures
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/jamesng1992/MOPSO-MONLTA-GlassMelter.git
cd MOPSO-MONLTA-GlassMelter
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

## Quick Start

### Run from command line

```bash
python scripts/run_pid_optimization.py
python scripts/run_pid_optimization.py --n-particles 40 --n-iterations 80
```

### Use in notebooks

```bash
cd notebooks
jupyter notebook 01_pid_tuning.ipynb
```

### Use as a library

```python
from mopso_monlta.optimizers import MOPSO, MONLTA
from mopso_monlta.evaluation import evaluate_pid_objectives, select_best_compromise

mopso = MOPSO(evaluate_pid_objectives, bounds=[(0.1,20),(0.001,5),(0,10)],
              n_particles=30, n_objectives=3)
mopso.optimize(n_iterations=60)

best_gains, best_obj = select_best_compromise(
    mopso.archive_positions, mopso.archive_objectives
)
```

## Notebooks

### 1. `01_pid_tuning.ipynb` — Core PID Tuning Comparison

Runs MOPSO, NTA, and MONLTA on the glass melter level control problem:
- Open-loop baseline response
- Pareto front visualization (2D & 3D)
- TOPSIS-based best compromise selection
- Closed-loop step response comparison
- Convergence analysis and MONLTA diagnostics

### 2. `02_advanced_applications.ipynb` — Extended Applications

Applies the framework to five additional glass melter control challenges:

| # | Application | Decision Variables | Objectives |
|---|-------------|-------------------|------------|
| 1 | **Neural ODE Hyperparameter Optimization** | learning rate, hidden size, epochs | Val. MSE, Training Time, Complexity |
| 2 | **Observer Gain Tuning** | $L_1, \ldots, L_4$ (Luenberger gains) | Est. Error, Noise Sensitivity, Settling |
| 3 | **MPC Weight Selection** | $Q$, $R$, $N$ | Tracking Error, Control Effort, Constraint Violation |
| 4 | **FOPID Controller Design** | $K_p, K_i, K_d, \lambda, \mu$ | IAE, Overshoot, Settling Time |
| 5 | **Multi-Zone Temperature PID** | 3×$(K_p, K_i, K_d)$ = 9 vars | Zone 1/2/3 IAE |

## Key MONLTA Features

- **Nonlinear Accepting Function**: $H(\zeta) = 1/\sqrt{1+(\zeta/\zeta_0)^2}$ — low-pass-filter form for controlled exploration-to-exploitation transition
- **Four Acceptance Scenarios**: Dominance-based acceptance with amount-of-domination principle
- **Variable-Size Archive**: Non-dominated solutions stored and updated using dominance rules
- **Focused Perturbation**: One decision variable perturbed at a time for targeted neighborhood search

## Key References

1. Nahas, Abouheaf, Darghouth, Sharaf, "A multi-objective AVR-LFC optimization scheme for multi-area power systems," *Electric Power Systems Research*, 200, 107467 (2021).
2. Coello et al., "Handling Multiple Objectives with Particle Swarm Optimization," *IEEE Transactions on Evolutionary Computation*, 2004.
3. Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II," *IEEE Transactions on Evolutionary Computation*, 2002.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

James Ng — [GitHub](https://github.com/jamesng1992)
