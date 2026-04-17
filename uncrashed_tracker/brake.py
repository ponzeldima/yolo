"""Гальмування дрона через Optical Flow.

Аналізує зсув сцени між кадрами і генерує протилежні
команди на стіки, щоб дрон завис на місці.

Декомпозиція optical flow на 3 осі руху:
  X (вліво/вправо) → рівномірний горизонтальний зсув → Roll
  Y (вверх/вниз)   → рівномірний вертикальний зсув   → Throttle
  Z (вперед/назад) → радіальне розходження/сходження  → Pitch
"""

import cv2
import numpy as np
import time

from . import config
from .pid import PIDController


class OpticalFlowBrake:
    """Оцінка руху дрона через dense optical flow + PID-корекція."""

    def __init__(self):
        self.pid_roll = PIDController(
            config.BRAKE_ROLL_KP, config.BRAKE_ROLL_KI, config.BRAKE_ROLL_KD,
            config.BRAKE_OUTPUT_MAX)
        self.pid_thr = PIDController(
            config.BRAKE_THR_KP, config.BRAKE_THR_KI, config.BRAKE_THR_KD,
            config.BRAKE_OUTPUT_MAX)
        self.pid_pitch = PIDController(
            config.BRAKE_PITCH_KP, config.BRAKE_PITCH_KI, config.BRAKE_PITCH_KD,
            config.BRAKE_OUTPUT_MAX)

        self._prev_gray: np.ndarray | None = None
        self._prev_time: float | None = None
        self._frame_time: float | None = None  # для dt між кадрами (EMA)
        self._radial_map: np.ndarray | None = None  # кешовані unit-вектори від центру
        self._smooth_roll = 0.0
        self._smooth_thr = -1.0 + config.BRAKE_BASE_THROTTLE * 2.0
        self._smooth_pitch = 0.0

        # Останні виміри (для HUD)
        self.flow_x = 0.0   # lateral (Roll)
        self.flow_y = 0.0   # vertical (Throttle)
        self.flow_div = 0.0  # divergence (Pitch / forward-back)

    def reset(self):
        self.pid_roll.reset()
        self.pid_thr.reset()
        self.pid_pitch.reset()
        self._prev_gray = None
        self._prev_time = None
        self._frame_time = None
        self._radial_map = None
        self._smooth_roll = 0.0
        self._smooth_thr = -1.0 + config.BRAKE_BASE_THROTTLE * 2.0
        self._smooth_pitch = 0.0
        self.flow_x = 0.0
        self.flow_y = 0.0
        self.flow_div = 0.0

    def _build_radial_map(self, h: int, w: int) -> np.ndarray:
        """Будує карту unit-векторів від центру зображення.

        Для кожного пікселя (x, y) обчислює нормалізований вектор
        від центру зображення до пікселя. Використовується для
        обрахунку дивергенції (рух вперед/назад).
        """
        cx, cy = w / 2.0, h / 2.0
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = xs - cx
        dy = ys - cy
        mag = np.sqrt(dx ** 2 + dy ** 2)
        mag[mag < 1e-6] = 1e-6  # уникаємо ділення на 0 в центрі
        # (H, W, 2) — unit-вектори від центру
        radial = np.stack([dx / mag, dy / mag], axis=-1)
        return radial

    @staticmethod
    def _time_alpha(alpha_per_sec: float, dt: float) -> float:
        """EMA-коефіцієнт, незалежний від FPS.

        alpha_per_sec: частка наближення до цілі за 1 секунду (0..1).
          0 = не змінюється, 1 = миттєво.
        dt: реальний час між кадрами (секунди).
        """
        if alpha_per_sec <= 0.0:
            return 0.0
        if alpha_per_sec >= 1.0:
            return 1.0
        return 1.0 - (1.0 - alpha_per_sec) ** dt

    def compute(self, frame: np.ndarray) -> tuple[float, float, float, float]:
        """Обчислює команди гальмування на основі optical flow.

        Returns:
            (left_x, left_y, right_x, right_y) — значення стіків.
        """
        h, w = frame.shape[:2]

        # Зменшуємо для швидкості
        scale = config.BRAKE_FLOW_SCALE
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_time = time.perf_counter()
            self._frame_time = self._prev_time
            base_thr = -1.0 + config.BRAKE_BASE_THROTTLE * 2.0
            return (0.0, base_thr, 0.0, 0.0)

        now = time.perf_counter()
        frame_dt = now - self._frame_time
        self._frame_time = now
        if frame_dt < 1e-6:
            frame_dt = 1e-6
        flow_dt = now - self._prev_time

        # Затухання кожен кадр (часо-незалежне)
        a_roll = self._time_alpha(config.BRAKE_ROLL_SMOOTH_ALPHA, frame_dt)
        a_thr = self._time_alpha(config.BRAKE_THR_SMOOTH_ALPHA, frame_dt)
        a_pitch = self._time_alpha(config.BRAKE_PITCH_SMOOTH_ALPHA, frame_dt)
        # Між обрахунками flow — затухання: roll/pitch до 0, thr до base
        self._smooth_roll = self._smooth_roll * (1 - a_roll)
        base_thr = -1.0 + config.BRAKE_BASE_THROTTLE * 2.0
        self._smooth_thr = base_thr + (self._smooth_thr - base_thr) * (1 - a_thr)
        self._smooth_pitch = self._smooth_pitch * (1 - a_pitch)

        # Обраховуємо flow лише кожні BRAKE_INTERVAL секунд
        if flow_dt < config.BRAKE_INTERVAL:
            return (
                0.0,
                max(-1.0, min(1.0, self._smooth_thr)),
                self._smooth_roll,
                self._smooth_pitch,
            )

        self._prev_time = now

        # Dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0,
        )
        self._prev_gray = gray

        # Центральна область (уникаємо країв і HUD)
        sh, sw = gray.shape[:2]
        margin_x = int(sw * 0.15)
        margin_y = int(sh * 0.15)
        roi = flow[margin_y:sh - margin_y, margin_x:sw - margin_x]

        # ── Кешована карта радіальних unit-векторів для ROI ──
        rh, rw = roi.shape[:2]
        if self._radial_map is None or self._radial_map.shape[:2] != (rh, rw):
            self._radial_map = self._build_radial_map(rh, rw)

        # ── Декомпозиція flow на 3 компоненти ──

        # 1) X-axis: рівномірний горизонтальний зсув → Roll
        mean_fx = float(np.mean(roi[..., 0]))

        # 2) Y-axis: рівномірний вертикальний зсув → Throttle
        mean_fy = float(np.mean(roi[..., 1]))

        # 3) Z-axis: дивергенція (dot product flow з radial unit-вектором)
        #    Позитивна = розходження від центру = рух вперед
        #    Негативна = сходження до центру = рух назад
        divergence = float(np.mean(
            roi[..., 0] * self._radial_map[..., 0] +
            roi[..., 1] * self._radial_map[..., 1]
        ))

        # Нормалізація: px/кадр → px/сек → приблизно [-1, +1]
        diag = np.sqrt(sw ** 2 + sh ** 2)
        norm_fx = (mean_fx / flow_dt) / (sw * config.BRAKE_FLOW_NORM)
        norm_fy = (mean_fy / flow_dt) / (sh * config.BRAKE_FLOW_NORM)
        norm_div = (divergence / flow_dt) / (diag * config.BRAKE_FLOW_NORM)

        self.flow_x = norm_fx
        self.flow_y = norm_fy
        self.flow_div = norm_div

        # ── PID: приводимо кожну компоненту до нуля ──

        # Roll: пікселі йдуть вправо (flow_x > 0) → дрон летить вліво → roll вправо
        roll_cmd = self.pid_roll.update(norm_fx)

        # Throttle: пікселі йдуть вниз (flow_y > 0) → дрон летить вверх → менше газу
        thr_correction = self.pid_thr.update(norm_fy)

        # Pitch: дивергенція > 0 → дрон летить вперед → pitch назад (інвертуємо)
        pitch_cmd = -self.pid_pitch.update(norm_div)

        base_thr = -1.0 + config.BRAKE_BASE_THROTTLE * 2.0
        throttle = base_thr - thr_correction

        # Roll і Throttle оновлюються вільно, Pitch — лише коли затух до ~0
        self._smooth_roll = roll_cmd
        self._smooth_thr = throttle
        if abs(self._smooth_pitch) < 0.01:
            self._smooth_pitch = pitch_cmd

        return (
            0.0,  # left_x (yaw) — не коригуємо
            max(-1.0, min(1.0, self._smooth_thr)),  # left_y (throttle)
            self._smooth_roll,   # right_x (roll)
            self._smooth_pitch,  # right_y (pitch)
        )
