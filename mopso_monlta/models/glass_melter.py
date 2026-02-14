"""
Glass melting furnace dynamics models.

Provides two model fidelities:

1. **2-state simplified model** (h, v):
   Fast enough for optimization inner loops (~15,000+ evaluations).
   Used for PID gain tuning in Notebook 1.

2. **7-state full model** (h, v, q_m, z1, z2, z3, z4):
   Includes Erlang-4 transport delay chain and melting lag.
   Used for the five advanced applications in Notebook 2.

Reference:
    Glass Melter Model Design (OI consortium)
"""

import numpy as np
from scipy.integrate import odeint

from ..config import GlassMelterParams2State, GlassMelterParams7State


# ============================================================
# 2-State Simplified Model
# ============================================================

def glass_melter_dynamics_2state(state, t, u_func, q_p_func, params):
    """
    Right-hand side of the simplified 2-state glass melter ODE.

    Equations:
        dh/dt = v
        dv/dt = (-v + K_m * (u - q_p)) / tau_m

    Args:
        state: [h, v] — current level [m] and rate of change [m/h]
        t: current time [h]
        u_func: callable(t) → batch charging rate [t/h]
        q_p_func: callable(t) → production pull disturbance [m³/h]
        params: GlassMelterParams2State instance

    Returns:
        [dh/dt, dv/dt]
    """
    h, v = state
    u = u_func(t)
    q_p = q_p_func(t)

    dh_dt = v
    dv_dt = (-v + params.K_m * (u - q_p)) / params.tau_m

    return [dh_dt, dv_dt]


# ============================================================
# 7-State Full Model
# ============================================================

def glass_melter_ode_7state(y, t, u1_func, u2_func, p):
    """
    7-state glass melter ODE right-hand side.

    States:
        y = [h, v, q_m, z1, z2, z3, z4]
        h    — glass level [m]
        v    — level velocity dh/dt [m/h]
        q_m  — molten glass flow rate [m³/h]
        z1–z4 — Erlang delay chain states [t/h]

    Signal flow:
        u1 → [z1→z2→z3→z4] → [Melting lag] → q_m → [Level] → h
                                                 ↑
                                                -u2 (pull)

    Args:
        y: state vector (7,)
        t: current time [h]
        u1_func: callable(t) → charging rate [t/h]
        u2_func: callable(t) → pull rate [m³/h]
        p: GlassMelterParams7State instance

    Returns:
        dy/dt (7,)
    """
    h, v, q_m, z1, z2, z3, z4 = y
    u1 = u1_func(t)
    u2 = u2_func(t)

    # Transport delay chain (Erlang N=4, mean delay = θ)
    a = p.N_delay / p.theta
    dz1 = a * (u1 - z1)
    dz2 = a * (z1 - z2)
    dz3 = a * (z2 - z3)
    dz4 = a * (z3 - z4)

    # Melting lag
    qm_ss = p.kc * z4
    dqm = (-q_m + qm_ss) / p.tau_m

    # Level dynamics
    dh = v
    dv = ((q_m - u2) / p.A - v) / p.tau_l

    return [dh, dv, dqm, dz1, dz2, dz3, dz4]


def simulate_glass_melter(u1_func, u2_func, y0, T_sim=20.0, dt=0.01, p=None):
    """
    Simulate the full 7-state glass melter using scipy's LSODA integrator.

    Args:
        u1_func: callable u1(t) — charging rate [t/h]
        u2_func: callable u2(t) — pull rate [m³/h]
        y0: initial state vector (7,)
        T_sim: simulation duration [h]
        dt: output time step [h]
        p: GlassMelterParams7State instance (default: nominal)

    Returns:
        t: time array [h]
        sol: state array (n_steps × 7)
    """
    if p is None:
        p = GlassMelterParams7State()
    t = np.arange(0, T_sim, dt)
    sol = odeint(glass_melter_ode_7state, y0, t, args=(u1_func, u2_func, p))
    return t, sol
