"""
Physical parameters and constants for the glass melting furnace.

Provides two parameter sets:
    - GlassMelterParams2State: Simplified 2-state model (h, v) for fast PID tuning
    - GlassMelterParams7State: Full 7-state model matching OI consortium notebooks

References:
    Glass Melter Model Design document (OI consortium)
    Open_Loop_Level_Modeling_NeuralODEs.ipynb
"""

from dataclasses import dataclass


# ============================================================
# 2-State Simplified Model Parameters
# ============================================================

@dataclass
class GlassMelterParams2State:
    """
    Parameters for the simplified 2-state glass melter model.

    This model captures the essential level dynamics with only two states
    (h, v), making it suitable for rapid PID gain evaluation during
    optimization sweeps (~15,000+ function evaluations).

    Attributes:
        tau_m: Melting time constant [h]
        K_m:   Melting gain [m³/t]
        A_tank: Tank cross-sectional area [m²]
        h_setpoint: Level setpoint [m]
        q_p_nom: Nominal production pull [m³/h]
    """
    tau_m: float = 2.0
    K_m: float = 0.15
    A_tank: float = 50.0
    h_setpoint: float = 0.9
    q_p_nom: float = 5.0


# ============================================================
# 7-State Full Model Parameters
# ============================================================

@dataclass
class GlassMelterParams7State:
    """
    Parameters for the full 7-state glass melter ODE model.

    States: x = [h, v, q_m, z1, z2, z3, z4]^T

    The model includes:
        - Erlang-4 transport delay chain (z1..z4)
        - First-order melting lag (q_m)
        - Second-order level dynamics (h, v)

    Reference: Glass Melter Model Design (OI consortium).

    Attributes:
        A:       Melt surface area [m²]
        rho:     Molten glass density [kg/m³]
        eta:     Batch-to-melt yield [-]
        kc:      Batch-to-flow gain [m³/h per t/h] = (eta × 1000) / rho
        tau_m:   Melting first-order time constant [h]
        theta:   Total transport delay [h]
        N_delay: Number of Erlang delay stages [-]
        tau_l:   Level dynamics lag [h]
        w0:      Nominal water flow [m³/h]
        kw:      Water flow sensitivity [1/(m³/h)]
        kp:      Pull calibration factor [-]
        h_setpoint: Nominal level setpoint [m]
        q_p_nom: Nominal production pull rate [m³/h]
    """
    A: float = 60.0
    rho: float = 2400.0
    eta: float = 0.95
    kc: float = 0.3958
    tau_m: float = 3.0
    theta: float = 2.0
    N_delay: int = 4
    tau_l: float = 0.10
    w0: float = 0.20
    kw: float = 0.02
    kp: float = 1.0
    h_setpoint: float = 0.9
    q_p_nom: float = 5.0


# ============================================================
# Constants
# ============================================================

# Glass density for unit conversion
GLASS_DENSITY = 2.4  # [tonnes/m³]

# State and input column names
STATE_COLS_7 = ["h", "v", "q_m", "z1", "z2", "z3", "z4"]
INPUT_COLS = ["u1", "u2", "w"]

# Default PID gain search bounds for optimization
DEFAULT_PID_BOUNDS = [
    (1.0, 500.0),    # Kp — proportional gain
    (0.1, 100.0),    # Ki — integral gain
    (0.0, 50.0),     # Kd — derivative gain
]

# Default simulation parameters
DEFAULT_T_SIM = 20.0          # [h] Total simulation time
DEFAULT_DT = 0.01             # [h] Integration time step
DEFAULT_H0 = 0.85             # [m] Initial level (below setpoint)
DEFAULT_SETPOINT = 0.9        # [m] Level setpoint
DEFAULT_DISTURBANCE_TIME = 10.0  # [h] When +20% pull disturbance occurs
