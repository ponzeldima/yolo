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
from collections import deque

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

        # Temporal median буфери
        n = config.BRAKE_MEDIAN_SIZE
        self._hist_fx: deque[float] = deque(maxlen=n)
        self._hist_fy: deque[float] = deque(maxlen=n)
        self._hist_div: deque[float] = deque(maxlen=n)

        # Auto-tune base throttle
        self._base_throttle = config.BRAKE_BASE_THROTTLE
        self._autotune_log: list[tuple[float, float, float]] = []  # (timestamp, throttle, flow_y)
        self._autotune_last_check: float | None = None
        self._gyro_missing_since: float | None = None
        self._gyro_last_warn: float | None = None

        # Цільова дивергенція (forward speed setpoint).
        # > 0 — політ вперед (пікселі мають розходитись із центру зі сталим темпом)
        # < 0 — політ назад
        # = 0 — зависання на місці
        self.forward_setpoint: float = 0.0

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
        self._hist_fx.clear()
        self._hist_fy.clear()
        self._hist_div.clear()
        self._base_throttle = config.BRAKE_BASE_THROTTLE
        self._autotune_log = []
        self._autotune_last_check = None
        self._gyro_missing_since = None
        self._gyro_last_warn = None

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

    def compute(self, frame: np.ndarray, gyro_pitch: float | None = None) -> tuple[float, float, float, float]:
        """Обчислює команди гальмування на основі optical flow.

        Args:
            frame: поточний кадр камери
            gyro_pitch: реальний pitch від гіроскопа дрона (градуси), або None

        Returns:
            (left_x, left_y, right_x, right_y) — значення стіків.
        """
        h, w = frame.shape[:2]

        # Зменшуємо для швидкості
        scale = config.BRAKE_FLOW_SCALE
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Gaussian blur — прибирає шум аналогової камери та jitter
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_time = time.perf_counter()
            self._frame_time = self._prev_time
            base_thr = -1.0 + self._base_throttle * 2.0
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
        base_thr = -1.0 + self._base_throttle * 2.0
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

        # Компенсація нахилу камери з урахуванням поточного pitch
        if gyro_pitch is not None:
            if self._gyro_missing_since is not None:
                gap = now - self._gyro_missing_since
                print(f"\n[BRAKE][GYRO] Telemetry restored after {gap:.2f}s (pitch={gyro_pitch:+.1f})")
                self._gyro_missing_since = None
                self._gyro_last_warn = None
            effective_comp = config.BRAKE_CAM_TILT_COMP - gyro_pitch * config.BRAKE_CAM_TILT_GYRO_FACTOR
        else:
            if self._gyro_missing_since is None:
                self._gyro_missing_since = now
                self._gyro_last_warn = now
                print("\n[BRAKE][GYRO] Telemetry missing -> fallback to smooth_pitch compensation")
            elif self._gyro_last_warn is None or (now - self._gyro_last_warn) >= 1.0:
                age = now - self._gyro_missing_since
                print(f"\n[BRAKE][GYRO] Telemetry still missing ({age:.1f}s), fallback active")
                self._gyro_last_warn = now
            effective_comp = config.BRAKE_CAM_TILT_COMP - self._smooth_pitch * config.BRAKE_CAM_TILT_PITCH_FACTOR
        norm_fy -= norm_div * effective_comp

        # Dead zone: шум аналогової камери нижче порогу → 0
        dz = config.BRAKE_DEAD_ZONE
        if abs(norm_fx) < (dz/10):
            norm_fx = 0.0
        if abs(norm_fy) < dz:
            norm_fy = 0.0
        if abs(norm_div) < (dz/10):
            norm_div = 0.0

        # Temporal median: робастний до викидів
        self._hist_fx.append(norm_fx)
        self._hist_fy.append(norm_fy)
        self._hist_div.append(norm_div)
        norm_fx = float(np.median(self._hist_fx))
        norm_fy = float(np.median(self._hist_fy))
        norm_div = float(np.median(self._hist_div))

        self.flow_x = norm_fx
        self.flow_y = norm_fy
        self.flow_div = norm_div

        # ── PID: приводимо кожну компоненту до нуля ──

        # Roll: пікселі йдуть вправо (flow_x > 0) → дрон летить вліво → roll вправо
        roll_cmd = self.pid_roll.update(norm_fx)

        # Throttle: пікселі йдуть вниз (flow_y > 0) → дрон летить вверх → менше газу
        thr_correction = self.pid_thr.update(norm_fy)

        # Pitch: дивергенція > 0 → дрон летить вперед → pitch назад (інвертуємо).
        # Віднімаємо setpoint: при forward_setpoint>0 PID буде підтримувати
        # стабільний forward-рух (дивергенція = setpoint вважається "нормою").
        pitch_cmd = -self.pid_pitch.update(norm_div - self.forward_setpoint)

        base_thr = -1.0 + self._base_throttle * 2.0
        throttle = base_thr - thr_correction

        # ── Auto-tune base throttle ──
        # Записуємо кожен вимір (timestamp, throttle_stick, flow_y)
        self._autotune_log.append((now, max(-1.0, min(1.0, throttle)), norm_fy))

        # Чистимо старші за вікно записи
        window = config.BRAKE_AUTOTUNE_WINDOW
        cutoff = now - window
        while self._autotune_log and self._autotune_log[0][0] < cutoff:
            self._autotune_log.pop(0)

        # Перевірка кожні AUTOTUNE_PERIOD секунд
        if self._autotune_last_check is None:
            self._autotune_last_check = now
        elif now - self._autotune_last_check >= config.BRAKE_AUTOTUNE_PERIOD:
            self._autotune_last_check = now
            # Потрібно мінімум window секунд даних
            if self._autotune_log and (now - self._autotune_log[0][0]) >= window * 0.9:
                thr_vals = [v[1] for v in self._autotune_log]
                fy_vals = [v[2] for v in self._autotune_log]
                thr_std = float(np.std(thr_vals))
                fy_std = float(np.std(fy_vals))

                # Відхиляємо якщо сильні скачки
                if thr_std < config.BRAKE_AUTOTUNE_MAX_THR_STD and \
                   fy_std < config.BRAKE_AUTOTUNE_MAX_FY_STD:
                    avg_thr = float(np.mean(thr_vals))
                    avg_fy = float(np.mean(fy_vals))
                    # Корекція: flow_y > 0 = дрон піднімається → менше газу
                    correction = avg_fy * config.BRAKE_AUTOTUNE_COEFF
                    new_thr_stick = avg_thr - correction
                    new_base = (max(-1.0, min(1.0, new_thr_stick)) + 1.0) / 2.0
                    if abs(new_base - self._base_throttle) > 0.002:
                        print(f"\n[BRAKE] Auto-tune: {self._base_throttle:.3f} → {new_base:.3f}"
                              f"  (avg_fy={avg_fy:+.3f}, fy_std={fy_std:.3f}, thr_std={thr_std:.3f})")
                        self._base_throttle = new_base
                else:
                    print(f"\n[BRAKE] Auto-tune skip: thr_std={thr_std:.3f}, fy_std={fy_std:.3f} (turbulence)")

        # Roll і Throttle оновлюються вільно, Pitch — hold: тримає значення коли div=0
        # if norm_fx != 0.0:
        self._smooth_roll = roll_cmd
        self._smooth_thr = throttle
        # if norm_div != 0.0:
        self._smooth_pitch = pitch_cmd

        return (
            0.0,  # left_x (yaw) — не коригуємо
            max(-1.0, min(1.0, self._smooth_thr)),  # left_y (throttle)
            self._smooth_roll,   # right_x (roll)
            self._smooth_pitch,  # right_y (pitch)
        )
