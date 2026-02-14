"""
MOPSO-MONLTA Glass Melter Optimization
=======================================

Multi-objective optimization framework for glass melting furnace control,
implementing MOPSO, NTA, and MONLTA algorithms.

Modules:
    config          - Plant parameters and physical constants
    models          - Glass melter dynamics and PID controller
    optimizers      - MOPSO, MONLTA, NTA (weighted-sum) algorithms
    evaluation      - Objective functions, Pareto utilities, decision-making
    visualization   - Publication-quality plotting
"""

__version__ = "0.1.0"
