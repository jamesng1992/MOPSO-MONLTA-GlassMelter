"""
PID Controller with anti-windup protection.

Implements a discrete-time PID controller with back-calculation
anti-windup to prevent integral saturation when the actuator
hits its physical limits.
"""

import numpy as np


class PIDController:
    """
    Discrete-time PID Controller with anti-windup.

    Control law:
        u = Kp * e + Ki * ∫e dt + Kd * de/dt

    Anti-windup: when the output saturates at [u_min, u_max],
    the integral term is back-calculated to prevent wind-up.

    Args:
        Kp: Proportional gain — immediate response to error
        Ki: Integral gain — eliminates steady-state error
        Kd: Derivative gain — provides damping
        u_min: Lower actuator limit [t/h]
        u_max: Upper actuator limit [t/h]
        dt: Time step [h]
    """

    def __init__(self, Kp, Ki, Kd, u_min=0, u_max=50, dt=0.01):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.u_min = u_min
        self.u_max = u_max
        self.dt = dt
        self.reset()

    def reset(self):
        """Reset integrator and derivative memory."""
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error):
        """
        Compute one step of PID control.

        Args:
            error: e(t) = setpoint - measurement

        Returns:
            u_sat: saturated control output ∈ [u_min, u_max]
        """
        # Proportional
        P = self.Kp * error

        # Integral (rectangular integration)
        self.integral += error * self.dt
        I = self.Ki * self.integral

        # Derivative (backward difference)
        D = self.Kd * (error - self.prev_error) / self.dt
        self.prev_error = error

        # Sum and saturate
        u = P + I + D
        u_sat = np.clip(u, self.u_min, self.u_max)

        # Anti-windup: back-calculation
        if u != u_sat and self.Ki > 0:
            self.integral -= (u - u_sat) / self.Ki

        return u_sat
