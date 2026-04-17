"""Автонаведення: PID-контролер + фази атаки (ACRO-режим)."""

from . import config
from .pid import PIDController


class AutoPilot:
    """PID-based auto-aim з фазами атаки для ACRO-режиму дрона."""

    def __init__(self):
        self.pid_yaw = PIDController(
            config.PID_YAW_KP, config.PID_YAW_KI, config.PID_YAW_KD, config.PID_OUTPUT_MAX)
        self.pid_thr = PIDController(
            config.PID_THR_KP, config.PID_THR_KI, config.PID_THR_KD, config.PID_OUTPUT_MAX)
        self.pid_pitch = PIDController(
            config.PID_PITCH_KP, config.PID_PITCH_KI, config.PID_PITCH_KD, config.PID_PITCH_MAX)

        self._smooth_yaw = 0.0
        self._smooth_thr = -1.0  # стартуємо з 0 газу (стік -1 = мін газ)
        self._smooth_pitch = 0.0
        self.attack_phase = "SEARCH"

    def reset(self):
        self.pid_yaw.reset()
        self.pid_thr.reset()
        self.pid_pitch.reset()
        self.attack_phase = "SEARCH"

    def compute(self, err_x: float, err_y: float, bbox_ratio: float
                ) -> tuple[float, float, float, float]:
        """Обчислює команди керування.

        Args:
            err_x: нормалізована похибка X (-1..+1, >0 = ціль справа)
            err_y: нормалізована похибка Y (-1..+1, >0 = ціль знизу)
            bbox_ratio: розмір цілі відносно ширини екрану

        Returns:
            (left_x, left_y, right_x, right_y) — значення стіків
        """
        # --- Визначення фази атаки ---
        if bbox_ratio >= config.PHASE_TERMINAL_RATIO:
            self.attack_phase = "TERMINAL"
        elif bbox_ratio >= config.PHASE_ATTACK_RATIO:
            self.attack_phase = "ATTACK"
        else:
            self.attack_phase = "APPROACH"

        # --- PID: yaw для горизонтального наведення ---
        yaw_cmd = self.pid_yaw.update(err_x)

        # --- PID: throttle для вертикального наведення ---
        thr_correction = self.pid_thr.update(err_y)
        base_thr_stick = -1.0 + config.BASE_THROTTLE_NORM * 2.0
        throttle_raw = base_thr_stick - thr_correction
        throttle_raw = max(-1.0, min(1.0, throttle_raw))

        # --- PID: pitch rate (ACRO) ---
        pitch_pid = self.pid_pitch.update(err_y)

        if self.attack_phase == "TERMINAL":
            base_pitch = config.BASE_PITCH_RATE + config.PHASE_TERMINAL_PITCH_ADD
        elif self.attack_phase == "ATTACK":
            base_pitch = config.BASE_PITCH_RATE + config.PHASE_ATTACK_PITCH_ADD
        else:
            base_pitch = config.BASE_PITCH_RATE

        pitch_raw = base_pitch + pitch_pid
        pitch_raw = max(-config.PID_PITCH_MAX * 2, min(1.0, pitch_raw))

        # --- EMA-згладжування ---
        alpha = config.SMOOTH_ALPHA
        self._smooth_yaw = self._smooth_yaw * (1 - alpha) + yaw_cmd * alpha
        self._smooth_thr = self._smooth_thr * (1 - alpha) + throttle_raw * alpha
        self._smooth_pitch = self._smooth_pitch * (1 - alpha) + pitch_raw * alpha

        return (self._smooth_yaw, self._smooth_thr, 0.0, self._smooth_pitch)

    def get_hold_commands(self) -> tuple[float, float, float, float]:
        """Команди утримання курсу (ціль втрачена в AUTO)."""
        return (self._smooth_yaw, self._smooth_thr, 0.0, 0.0)
