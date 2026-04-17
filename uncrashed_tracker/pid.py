"""Дискретний PID-регулятор з anti-windup."""

import time


class PIDController:

    def __init__(self, kp: float, ki: float, kd: float, output_max: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_max = output_max
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def update(self, error: float) -> float:
        now = time.perf_counter()
        if self._prev_time is None:
            self._prev_time = now
            self._prev_error = error
            return self.kp * error  # перший кадр — тільки P

        dt = now - self._prev_time
        if dt < 1e-6:
            return 0.0
        self._prev_time = now

        # P
        p = self.kp * error
        # I з anti-windup
        self._integral += error * dt
        i_limit = self.output_max / max(self.ki, 1e-9)
        self._integral = max(-i_limit, min(i_limit, self._integral))
        i = self.ki * self._integral
        # D
        d = self.kd * (error - self._prev_error) / dt
        self._prev_error = error

        out = p + i + d
        return max(-self.output_max, min(self.output_max, out))
