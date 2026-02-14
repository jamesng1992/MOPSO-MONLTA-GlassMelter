"""
Closed-loop simulation and objective function evaluation.

Provides the "inner loop" functions called thousands of times by
each optimizer during PID gain tuning.

Performance metrics follow Nahas et al. (2021):
    IAE, ISE, ITSE, ITAE, overshoot, settling time, rise time, peak time.
"""

import numpy as np

from ..models.pid import PIDController
from ..config import (
    GlassMelterParams2State,
    GlassMelterParams7State,
    DEFAULT_T_SIM,
    DEFAULT_DT,
    DEFAULT_H0,
    DEFAULT_SETPOINT,
    DEFAULT_DISTURBANCE_TIME,
)


def simulate_closed_loop(Kp, Ki, Kd, params=None, T_sim=None, dt=None,
                         h0=None, setpoint=None, disturbance_time=None):
    """
    Simulate PID-controlled glass melter in closed loop.

    Uses a simplified 2-state model (h, v) for computational speed.

    Test scenario:
        Phase 1 (0 → disturbance_time): step tracking from h0 to setpoint
        Phase 2 (disturbance_time → T_sim): +20% production pull disturbance

    Args:
        Kp, Ki, Kd: PID controller gains
        params: GlassMelterParams2State or GlassMelterParams7State
        T_sim: total simulation time [h]
        dt: integration time step [h]
        h0: initial glass level [m]
        setpoint: desired glass level [m]
        disturbance_time: when +20% pull disturbance occurs [h]

    Returns:
        t: time array
        h: level trajectory
        u: control input trajectory
        metrics: dict with IAE, ISE, ITSE, ITAE, overshoot,
                 settling_time, rise_time, peak_time
    """
    # Apply defaults
    if params is None:
        params = GlassMelterParams2State()
    if T_sim is None:
        T_sim = DEFAULT_T_SIM
    if dt is None:
        dt = DEFAULT_DT
    if h0 is None:
        h0 = DEFAULT_H0
    if setpoint is None:
        setpoint = DEFAULT_SETPOINT
    if disturbance_time is None:
        disturbance_time = DEFAULT_DISTURBANCE_TIME

    n_steps = int(T_sim / dt)
    t = np.linspace(0, T_sim, n_steps)
    h = np.zeros(n_steps)
    v = np.zeros(n_steps)
    u = np.zeros(n_steps)
    h[0] = h0
    v[0] = 0.0

    pid = PIDController(Kp, Ki, Kd, u_min=0, u_max=50, dt=dt)

    # Extract effective gain depending on parameter type
    if isinstance(params, GlassMelterParams7State):
        K_m = params.kc / params.A
        tau_eff = params.tau_m
    else:
        K_m = params.K_m
        tau_eff = params.tau_m

    q_p_nom = params.q_p_nom

    # Forward Euler integration
    for i in range(1, n_steps):
        error = setpoint - h[i - 1]
        u[i] = pid.compute(error)
        q_p_val = q_p_nom * 1.2 if t[i] >= disturbance_time else q_p_nom
        dv = (-v[i - 1] + K_m * (u[i] - q_p_val)) / tau_eff
        v[i] = v[i - 1] + dv * dt
        h[i] = h[i - 1] + v[i] * dt

    # ---- Compute performance metrics ----
    err = setpoint - h
    abs_err = np.abs(err)
    sq_err = err ** 2
    step_size = abs(setpoint - h0)

    IAE = np.trapezoid(abs_err, t)
    ISE = np.trapezoid(sq_err, t)
    ITSE = np.trapezoid(t * sq_err, t)
    ITAE = np.trapezoid(t * abs_err, t)

    overshoot = (max(0, (np.max(h) - setpoint) / step_size * 100)
                 if h0 < setpoint else 0.0)

    # Settling time (2% band)
    tol = 0.02 * step_size
    settled = np.abs(h - setpoint) <= tol
    settling_time = T_sim
    for i in range(len(settled) - 1, -1, -1):
        if not settled[i]:
            settling_time = t[min(i + 1, len(t) - 1)]
            break

    # Rise time (10% → 90%)
    t10, t90 = T_sim, T_sim
    for i in range(len(h)):
        if h[i] >= h0 + 0.1 * step_size:
            t10 = t[i]
            break
    for i in range(len(h)):
        if h[i] >= h0 + 0.9 * step_size:
            t90 = t[i]
            break
    rise_time = t90 - t10

    # Peak time
    peak_time = t[np.argmax(h)]

    metrics = dict(
        IAE=IAE, ISE=ISE, ITSE=ITSE, ITAE=ITAE,
        overshoot=overshoot, settling_time=settling_time,
        rise_time=rise_time, peak_time=peak_time,
    )
    return t, h, u, metrics


def evaluate_pid_objectives(x, params=None):
    """
    Evaluate tri-objective PID tuning fitness: [IAE, Overshoot, Settling Time].

    This is the objective function each optimizer calls repeatedly.

    Args:
        x: [Kp, Ki, Kd] — candidate PID gains
        params: plant model parameters

    Returns:
        [IAE, overshoot, settling_time] (all minimized)
        [1e6, 1e6, 1e6] for invalid/unstable solutions
    """
    Kp, Ki, Kd = x

    if Kp <= 0 or Ki < 0 or Kd < 0:
        return [1e6, 1e6, 1e6]

    try:
        _, _, _, metrics = simulate_closed_loop(Kp, Ki, Kd, params)
        return [metrics['IAE'], metrics['overshoot'], metrics['settling_time']]
    except Exception:
        return [1e6, 1e6, 1e6]
