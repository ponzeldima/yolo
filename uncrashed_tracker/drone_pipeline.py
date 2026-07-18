"""Конвеєр для реального дрона: камера + трекінг + джойстік + автопілот → дрон.

Поєднує:
  - CameraSource (USB-камера) для відеопотоку
  - YOLO-трекер для детекції об'єктів
  - PhysicalJoystick для ручного керування
  - AutoPilot / OpticalFlowBrake для автоматичного керування
  - DroneSender для передачі команд на реальний дрон через USB-VCP

Режими (ідентичні DroneAimPipeline):
  MANUAL — passthrough джойстіка на дрон
  LOCK   — ціль залоковано, керування з джойстіка
  AUTO   — автопілот наводиться на ціль
  BRAKE  — optical flow гальмування (зависання)
"""

import time
import threading
from collections import deque

import cv2
import numpy as np
import keyboard

from . import config
from .models import Detection
from .image_source import BaseImageSource
from .tracker import BaseTracker
from .controller import PhysicalJoystick
from .drone_sender import DroneSender
from .autopilot import AutoPilot
from .brake import OpticalFlowBrake
from .hunt import HuntController
from . import visualizer


class DroneTrackingPipeline:
    """Камера → Трекінг → Джойстік/Автопілот → Реальний дрон."""

    WINDOW_NAME = "Drone Tracker"

    def __init__(self, image_source: BaseImageSource, tracker: BaseTracker,
                 drone_port: str):
        self.source = image_source
        self.tracker = tracker

        self.joystick = PhysicalJoystick()
        self.drone = DroneSender(drone_port)
        self.autopilot = AutoPilot()
        self.brake = OpticalFlowBrake()
        self.hunt: HuntController | None = None   # lazy init при першому активуванні
        self.hunt_active = False

        # ── Стан ──
        self.flight_mode = "MANUAL"
        self.locked_track_id = None
        self.closest_track_id = None
        self.lost_since = None
        self._quit_flag = False

        # ── Async inference ──
        self._infer_lock = threading.Lock()
        self._latest_detections: list[Detection] = []
        self._latest_frame: np.ndarray | None = None
        self._infer_fps = 0.0
        self._infer_seq = 0
        self._infer_running = False
        self._cam_fps = 0.0

        # ── Динамічний trim (bias) для roll/pitch ──
        # Зберігаємо останні TRIM_WINDOW_SEC фактично відправлених команд
        # разом з залишковим optical-flow (flow_x, flow_div).
        # Раз на TRIM_UPDATE_SEC обчислюємо:
        #     trim = avg(cmd) + K * avg(residual_flow)
        # Це дозволяє trim швидше сходитися до істинного значення: якщо PID видавав
        # середнє rx=0.10, але залишок flow_x все ще ненульовий — треба йти далі.
        self.TRIM_WINDOW_SEC = 2.0
        self.TRIM_UPDATE_SEC = 1000.0
        # Коефіцієнти резидуального flow → trim (різні для roll і pitch, бо різні одиниці:
        # flow_x — піксельний зсув, flow_div — коефіцієнт розбіжності; і PID у них різні Kp).
        self.TRIM_K_ROLL = 0.15    # trim_roll  += K_ROLL  * mean(flow_x)
        self.TRIM_K_PITCH = 0.30   # trim_pitch += K_PITCH * mean(flow_div)
        # Поріг зміни, при якому скидаємо I-term PID (щоб не обнуляти при мікро-поправках).
        self.TRIM_PID_RESET_THRESH = 0.02
        self.trim_roll = 0.012   # початкове значення
        self.trim_pitch = 0.09  # початкове значення
        # (timestamp, roll_cmd, pitch_cmd, flow_x, flow_div)
        self._trim_history: deque[tuple[float, float, float, float, float]] = deque()
        self._trim_last_update = 0.0

        # ── Forward-speed setpoint (w/s) ──
        # Крок зміни цільової дивергенції (в тих самих одиницях, що flow_div після нормалізації).
        # 0.05 — помірний приріст швидкості; діапазон обмежимо ±0.5.
        self.FORWARD_STEP = 0.05
        self.FORWARD_MAX = 0.5

        # ── Yaw-керування на клавіатурі (a/d, утримання) ──        # Поки натиснута клавіша — видаємо фіксовану yaw-команду.
        # На час повороту + COOLDOWN заморожуємо roll-контур (PID та trim-update),
        # бо панорамування камери дає паразитний flow_x, який інакше був би
        # інтерпретований як боковий дрейф.
        self.YAW_RATE_CMD = 0.4
        self.YAW_COOLDOWN = 0.9  # сек після останнього yaw-тіку (мануал або HUNT)
        self._yaw_left_pressed = False
        self._yaw_right_pressed = False
        # Час останнього кадру, де видавалась yaw-команда (мануал або HUNT).
        # Завдяки єдиний cooldown діє для обох джерел.
        self._yaw_last_active_at: float | None = None
        self._was_yawing = False  # для детекту фронту "кінець повороту"

        # ── Pitch → throttle feed-forward ──
        # Коли |pitch − trim_pitch| зростає, частина тяги йде вбік → дрон просідає.
        # Доливаємо газ пропорційно ПОХІДНІЙ відхилення (тільки під час зміни,
        # стабільний нахил не впливає — на стабільний уже реагує auto-tune base throttle).
        # Знак: зростає |dev| → +throttle; зменшується → −throttle.
        self.PITCH_THROTTLE_FF_K = 0.0   # коефіцієнт feed-forward (d|dev|/dt → приріст стіка)
        self.PITCH_THROTTLE_FF_MAX = 0.15  # максимальний миттєвий приріст стіка
        self._prev_pitch_dev_abs: float | None = None
        self._prev_pitch_ff_time: float | None = None

        # ── HUNT ATTACK: хардкодне значення газу ──
        # Під час тарану pitch отримує велике значення, від чого flow_y/div
        # «стрибають» і throttle-PID видає хаотичний газ. Тому в фазі ATTACK
        # жорстко тримаємо ly на фіксованому значенні [-1..+1] (не стікові 0..1,
        # а вже після конверсії set_sticks в міксер). 0.60 нормалізованих одиниць
        # = -1 + 0.60*2 = +0.20 на стіку.
        self.ATTACK_THROTTLE_NORM = 0.48  # 0..1 (0 = мін, 1 = макс)
        # Pitch у фазі ATTACK теж хардкодимо, бо pitch-PID вираховує його з
        # (flow_div - forward_setpoint), що в момент тарана дає хаотичні значення.
        # Діапазон: [-1..+1], + це вперед. 0.40 = сильний нахил на ціль.
        self.ATTACK_PITCH_STICK = 0.70

    # ── Стан-машина режимів ──────────────────────────────────────────────────

    def _switch_to_manual(self):
        self.flight_mode = "MANUAL"
        self.locked_track_id = None
        self.lost_since = None
        self.autopilot.reset()
        self.brake.reset()
        self.drone.reset()
        print("\n[MODE] MANUAL — джойстік → дрон напряму")

    def _toggle_mode(self, _event=None):
        if self.flight_mode == "BRAKE":
            return
        if self.flight_mode == "MANUAL":
            if self.closest_track_id is not None:
                self.flight_mode = "LOCK"
                self.locked_track_id = self.closest_track_id
                print(f"\n[MODE] LOCK — ціль ID={self.locked_track_id}")
            else:
                print("\n[НЕМАЄ ЦІЛІ] Немає об'єкта для локу")
        elif self.flight_mode == "LOCK":
            self.flight_mode = "AUTO"
            self.autopilot.reset()
            self.autopilot.attack_phase = "APPROACH"
            print(f"\n[MODE] AUTO — наведення на ID={self.locked_track_id}")
        elif self.flight_mode == "AUTO":
            self._switch_to_manual()

    def _toggle_brake(self, _event=None):
        if self.flight_mode == "BRAKE":
            self._switch_to_manual()
        else:
            self.flight_mode = "BRAKE"
            self.locked_track_id = None
            self.lost_since = None
            self.autopilot.reset()
            self.brake.reset()
            self._reset_trim()
            self._reset_pitch_throttle_ff()
            self.brake.forward_setpoint = 0.0
            self.hunt_active = False
            if self.hunt is not None:
                self.hunt.reset()
            self._brake_start_time = time.perf_counter()
            print("\n[MODE] BRAKE — optical flow зависання (зліт 0.5с)")

    def _toggle_hunt(self, _event=None):
        """Увімкнення/вимкнення автопошуку людини в BRAKE."""
        if self.flight_mode != "BRAKE":
            print("\n[HUNT] Активація лише в BRAKE")
            return
        if self.hunt is None:
            print("\n[HUNT] Завантаження YOLOv8s…")
            self.hunt = HuntController(
                model_path="yolov8s.pt",
                conf=config.HUNT_CONFIDENCE_THRESHOLD,
            )
            print("[HUNT] YOLOv8s готовий")
        if self.hunt_active:
            self.hunt_active = False
            self.hunt.reset()
            self.brake.forward_setpoint = 0.0
            self.brake.pid_pitch.reset()
            print("\n[HUNT] ВИМКНЕНО")
        else:
            self.hunt.reset()
            self.hunt_active = True
            print("\n[HUNT] УВІМКНЕНО — SCAN → ALIGN → ATTACK")

    def _forward_speed_up(self, _event=None):
        """Збільшує цільову forward-швидкість (дрон поступово летить вперед)."""
        if self.flight_mode != "BRAKE":
            return
        self.brake.forward_setpoint = min(
            self.FORWARD_MAX,
            self.brake.forward_setpoint + self.FORWARD_STEP,
        )
        self.brake.pid_pitch.reset()
        print(f"\n[BRAKE] forward_setpoint = {self.brake.forward_setpoint:+.2f}")

    def _forward_speed_down(self, _event=None):
        """Зменшує цільову forward-швидкість (до 0 — зависання, нижче — назад)."""
        if self.flight_mode != "BRAKE":
            return
        self.brake.forward_setpoint = max(
            -self.FORWARD_MAX,
            self.brake.forward_setpoint - self.FORWARD_STEP,
        )
        self.brake.pid_pitch.reset()
        print(f"\n[BRAKE] forward_setpoint = {self.brake.forward_setpoint:+.2f}")

    def _forward_speed_stop(self, _event=None):
        """Миттєво обнуляє forward-швидкість (повертає до зависання)."""
        if self.flight_mode != "BRAKE":
            return
        self.brake.forward_setpoint = 0.0
        self.brake.pid_pitch.reset()
        print("\n[BRAKE] forward_setpoint = 0.00 (hover)")

    # ── Yaw (a/d, утримання) ─────────────────────────────────────────────────

    def _yaw_left_down(self, _event=None):
        self._yaw_left_pressed = True

    def _yaw_left_up(self, _event=None):
        self._yaw_left_pressed = False

    def _yaw_right_down(self, _event=None):
        self._yaw_right_pressed = True

    def _yaw_right_up(self, _event=None):
        self._yaw_right_pressed = False

    def _current_yaw_cmd(self) -> float:
        """Yaw-стік від клавіатури (-1..+1, сума щоб обидві = 0)."""
        cmd = 0.0
        if self._yaw_left_pressed:
            cmd -= self.YAW_RATE_CMD
        if self._yaw_right_pressed:
            cmd += self.YAW_RATE_CMD
        return cmd

    def _resolve_yaw_cmd(self, hunt_yaw_cmd: float | None, hunt_yaw_scanning: bool) -> float:
        """Єдине джерело yaw-команди. Пріоритет: HUNT > клавіатура."""
        if hunt_yaw_cmd is not None and (abs(hunt_yaw_cmd) > 0.01 or hunt_yaw_scanning):
            return hunt_yaw_cmd
        return self._current_yaw_cmd()

    def _mark_yaw_active(self):
        """Фіксує момент останнього активного yaw-тіку (запускає cooldown)."""
        self._yaw_last_active_at = time.perf_counter()

    def _is_yaw_active(self) -> bool:
        """True якщо в останній yaw-тік (мануал або HUNT) був < YAW_COOLDOWN тому."""
        if self._yaw_last_active_at is None:
            return False
        return (time.perf_counter() - self._yaw_last_active_at) < self.YAW_COOLDOWN

    def _pitch_throttle_feedforward(self, out_pitch: float) -> float:
        """Компенсація висоти при зміні pitch.

        Повертає приріст до throttle-стіка (може бути від'ємним).
        Базується на похідній від |out_pitch − trim_pitch|:
          зростає → дрон нахиляється сильніше → просідає → +газ;
          зменшується → повертається до hover → надмір тяги → −газ;
          стабільно → 0.
        """
        now = time.perf_counter()
        dev_abs = abs(out_pitch - self.trim_pitch)

        if self._prev_pitch_dev_abs is None or self._prev_pitch_ff_time is None:
            self._prev_pitch_dev_abs = dev_abs
            self._prev_pitch_ff_time = now
            return 0.0

        dt = now - self._prev_pitch_ff_time
        if dt < 1e-4:
            return 0.0

        d_dev = (dev_abs - self._prev_pitch_dev_abs) / dt
        self._prev_pitch_dev_abs = dev_abs
        self._prev_pitch_ff_time = now

        ff = self.PITCH_THROTTLE_FF_K * d_dev
        # Клампінг, щоб раптовий стрибок не вистрілив газом
        ff = max(-self.PITCH_THROTTLE_FF_MAX, min(self.PITCH_THROTTLE_FF_MAX, ff))
        return ff

    def _reset_pitch_throttle_ff(self):
        self._prev_pitch_dev_abs = None
        self._prev_pitch_ff_time = None

    # ── Фоновий інференс ─────────────────────────────────────────────────────

    def _inference_loop(self):
        cam_frame_count = 0
        cam_fps_timer = time.perf_counter()
        while self._infer_running:
            frame = self.source.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            cam_frame_count += 1
            now = time.perf_counter()
            cam_elapsed = now - cam_fps_timer
            if cam_elapsed >= 1.0:
                self._cam_fps = cam_frame_count / cam_elapsed
                cam_frame_count = 0
                cam_fps_timer = now
            t = now
            # detections = self.tracker.track(frame)
            dt = time.perf_counter() - t
            fps = 1.0 / (dt + 1e-9)
            with self._infer_lock:
                # self._latest_detections = detections
                self._infer_fps = fps
                self._infer_seq += 1
                self._latest_frame = frame.copy()

    # ── Допоміжні ────────────────────────────────────────────────────────────

    @staticmethod
    def _pick_closest_to_center(detections: list[Detection], cx: int, cy: int) -> Detection | None:
        best = None
        best_dist = float('inf')
        for det in detections:
            dcx, dcy = det.center
            dist = (dcx - cx) ** 2 + (dcy - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best = det
        return best

    @staticmethod
    def _find_by_track_id(detections: list[Detection], track_id: int) -> Detection | None:
        for det in detections:
            if det.track_id == track_id:
                return det
        return None

    # ── Динамічний trim ──────────────────────────────────────────────────────

    def _update_trim(self, roll_cmd: float, pitch_cmd: float,
                     flow_x: float, flow_div: float):
        """Додає фактичну команду та залишковий flow у ковзне вікно (TRIM_WINDOW_SEC).
        Раз на TRIM_UPDATE_SEC оновлює trim за формулою:
            trim_roll  = mean(roll_cmd)  + K_ROLL  * mean(flow_x)
            trim_pitch = mean(pitch_cmd) + K_PITCH * mean(flow_div)
        Якщо стрибок trim > TRIM_PID_RESET_THRESH — скидає I-term відповідного PID,
        щоб накопичена поправка не дала overshoot.
        """
        now = time.perf_counter()
        self._trim_history.append((now, roll_cmd, pitch_cmd, flow_x, flow_div))
        # Обрізаємо старіше ніж TRIM_WINDOW_SEC
        cutoff = now - self.TRIM_WINDOW_SEC
        while self._trim_history and self._trim_history[0][0] < cutoff:
            self._trim_history.popleft()
        # Перерахунок раз на TRIM_UPDATE_SEC
        if now - self._trim_last_update >= self.TRIM_UPDATE_SEC and len(self._trim_history) >= 2:
            n = len(self._trim_history)
            avg_roll_cmd = sum(r for _, r, _, _, _ in self._trim_history) / n
            avg_pitch_cmd = sum(p for _, _, p, _, _ in self._trim_history) / n
            avg_flow_x = sum(fx for _, _, _, fx, _ in self._trim_history) / n
            avg_flow_div = sum(fd for _, _, _, _, fd in self._trim_history) / n

            new_trim_roll = avg_roll_cmd + self.TRIM_K_ROLL * avg_flow_x
            new_trim_pitch = avg_pitch_cmd + self.TRIM_K_PITCH * avg_flow_div

            d_roll = new_trim_roll - self.trim_roll
            d_pitch = new_trim_pitch - self.trim_pitch

            self.trim_roll = new_trim_roll
            self.trim_pitch = new_trim_pitch
            self._trim_last_update = now

            # Скидання I-term PID при стрибку trim (щоб старий інтеграл не наклався)
            if abs(d_roll) > self.TRIM_PID_RESET_THRESH:
                self.brake.pid_roll.reset()
            if abs(d_pitch) > self.TRIM_PID_RESET_THRESH:
                self.brake.pid_pitch.reset()

    def _reset_trim(self):
        self._trim_history.clear()
        self._trim_last_update = 0.0
        self.trim_roll = 0.012
        self.trim_pitch = 0.09

    # ── Візуалізація BRAKE ─────────────────────────────────────────────────

    def _draw_brake_overlay(self, frame: np.ndarray, cx: int, cy: int):
        """Малює коло в центрі + стрілки напрямку optical flow."""
        # Коло в центрі
        radius = 40
        cv2.circle(frame, (cx, cy), radius, (0, 255, 255), 2)

        # Стрілка lateral (flow_x) — горизонтальний зсув → Roll
        # Стрілка vertical (flow_y) — вертикальний зсув → Throttle
        arrow_scale = 150  # px на одиницю flow
        fx = self.brake.flow_x
        fy = self.brake.flow_y
        div = self.brake.flow_div

        # XY-стрілка від центру (cyan)
        dx = int(fx * arrow_scale)
        dy = int(fy * arrow_scale)
        if abs(dx) > 3 or abs(dy) > 3:
            cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy),
                            (0, 255, 255), 3, tipLength=0.25)

        # Дивергенція: додаткове коло (більше = вперед, менше = назад)
        div_radius = int(radius + div * 80)
        div_radius = max(10, min(200, div_radius))
        div_color = (0, 200, 0) if div > 0 else (0, 0, 255)
        cv2.circle(frame, (cx, cy), div_radius, div_color, 1)

        # Підписи осей зі значеннями
        cv2.putText(frame, f"X:{fx:+.2f}", (cx + radius + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Y:{fy:+.2f}", (cx + radius + 10, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"D:{div:+.2f}", (cx + radius + 10, cy + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, div_color, 1)

    # ── Головний цикл ────────────────────────────────────────────────────────

    def run(self):
        # Ініціалізація
        self.joystick.init()
        self.drone.init()

        _, _, width, height = self.source.get_region()
        cx_screen = width // 2
        cy_screen = height // 2

        # Клавіатура
        keyboard.on_press_key("space", self._toggle_mode)
        keyboard.on_press_key("b", self._toggle_brake)
        keyboard.on_press_key("w", self._forward_speed_up)
        keyboard.on_press_key("s", self._forward_speed_down)
        keyboard.on_press_key("x", self._forward_speed_stop)
        keyboard.on_press_key("a", self._yaw_left_down)
        keyboard.on_release_key("a", self._yaw_left_up)
        keyboard.on_press_key("d", self._yaw_right_down)
        keyboard.on_release_key("d", self._yaw_right_up)
        keyboard.on_press_key("h", self._toggle_hunt)
        keyboard.on_press_key("q", lambda _: setattr(self, '_quit_flag', True))

        # Фоновий інференс
        self._infer_running = True
        infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        infer_thread.start()

        print(f"[DRONE] Камера: {width}x{height}")
        print(f"[DRONE] Порт: {self.drone._port}")
        print("[DRONE] ПРОБІЛ — MANUAL → LOCK → AUTO → MANUAL")
        print("[DRONE] 'b' — BRAKE (optical flow зависання)")
        print("[DRONE] 'w'/'s' — forward speed +/-, 'x' — hover (0)")
        print("[DRONE] 'a'/'d' — yaw left/right (утримання)")
        print("[DRONE] 'h' — HUNT (autonomous person attack)")
        print("[DRONE] 'q' — вихід")
        print("[MODE] MANUAL — джойстік → дрон напряму\n")

        frame_count = 0

        try:
            while not self._quit_flag:
                t0 = time.perf_counter()
                frame_count += 1

                # Отримання детекцій
                with self._infer_lock:
                    detections = list(self._latest_detections)
                    seq = self._infer_seq
                    infer_fps = self._infer_fps
                    current_frame = self._latest_frame

                if current_frame is None:
                    if self.flight_mode == "MANUAL":
                        p_roll, p_pitch, p_throttle, p_yaw = self.joystick.read()
                        self.drone.set_sticks(p_yaw, p_throttle, p_roll, p_pitch)
                    self.drone.read_telemetry()
                    time.sleep(0.01)
                    continue

                # ── Телеметрія від дрона (pitch/roll) ──
                telem_pitch, telem_roll = self.drone.read_telemetry()

                # ── Вибір цілі ──
                target = None

                # if self.flight_mode == "AUTO":
                #     if self.locked_track_id is not None:
                #         target = self._find_by_track_id(detections, self.locked_track_id)
                #         if target:
                #             self.lost_since = None
                #         else:
                #             if self.lost_since is None:
                #                 self.lost_since = time.perf_counter()
                #             if (time.perf_counter() - self.lost_since) >= config.LOST_TIMEOUT:
                #                 print(f"\n[ВТРАТА] Ціль ID={self.locked_track_id} зникла → MANUAL")
                #                 self._switch_to_manual()

                # elif self.flight_mode == "LOCK":
                #     if self.locked_track_id is not None:
                #         target = self._find_by_track_id(detections, self.locked_track_id)
                #         if target:
                #             self.lost_since = None
                #         else:
                #             if self.lost_since is None:
                #                 self.lost_since = time.perf_counter()
                #             if (time.perf_counter() - self.lost_since) >= config.LOST_TIMEOUT:
                #                 print(f"\n[ВТРАТА] Ціль ID={self.locked_track_id} зникла → MANUAL")
                #                 self._switch_to_manual()

                # else:  # MANUAL / BRAKE
                #     closest = self._pick_closest_to_center(detections, cx_screen, cy_screen)
                #     if closest:
                #         self.closest_track_id = closest.track_id
                #         target = closest
                #     else:
                #         self.closest_track_id = None

                # ── Керування дроном ──
                if self.flight_mode == "BRAKE" and current_frame is not None:
                    brake_elapsed = time.perf_counter() - self._brake_start_time
                    if brake_elapsed < 2.0:
                        # Фаза зльоту: базовий газ, поточний trim по roll/pitch
                        base_thr = -1.0 + config.BRAKE_BASE_THROTTLE * 2.0
                        out_roll = self.trim_roll
                        out_pitch = self.trim_pitch
                        self.drone.set_sticks(0.0, base_thr, out_roll, out_pitch)
                    else:
                        # ── HUNT: автопошук і таран людини ──
                        # Викликається ДО brake.compute(), бо керує forward_setpoint.
                        hunt_yaw_cmd: float | None = None
                        hunt_yaw_scanning = False
                        if self.hunt_active and self.hunt is not None:
                            h_yaw, h_fwd, h_scanning = self.hunt.update(current_frame)
                            self.brake.forward_setpoint = h_fwd
                            hunt_yaw_cmd = h_yaw
                            hunt_yaw_scanning = h_scanning

                        lx, ly, rx, ry = self.brake.compute(current_frame, gyro_pitch=telem_pitch)

                        # Єдине джерело yaw-команди (HUNT > клавіатура).
                        yaw_cmd = self._resolve_yaw_cmd(hunt_yaw_cmd, hunt_yaw_scanning)
                        if abs(yaw_cmd) > 0.01:
                            self._mark_yaw_active()
                        yaw_active = self._is_yaw_active()

                        if yaw_active:
                            lx = yaw_cmd
                            # Під час yaw + cooldown ігноруємо roll-PID:
                            # панорамування камери дає паразитний flow_x → rx «брехливий».
                            self.brake.pid_roll.reset()
                            rx = 0.0
                            out_roll = self.trim_roll
                            out_pitch = ry + self.trim_pitch
                        else:
                            # Щойно закінчився поворот — скидаємо медіанні буфери flow,
                            # бо в них ще лежать «брехливі» значення від yaw-панорами.
                            if self._was_yawing:
                                self.brake._hist_fx.clear()
                                self.brake._hist_fy.clear()
                                self.brake.pid_roll.reset()
                                self.brake.pid_thr.reset()
                            out_roll = rx + self.trim_roll
                            out_pitch = ry + self.trim_pitch

                        self._was_yawing = yaw_active

                        # HUNT ATTACK: жорстко фіксуємо газ і pitch, щоб PID не розкачував їх
                        # від «брехливого» flow під час тарана.
                        if (self.hunt_active and self.hunt is not None
                                and self.hunt.phase == self.hunt.PHASE_ATTACK):
                            self.brake.pid_thr.reset()
                            self.brake.pid_pitch.reset()
                            ly = -1.0 + self.ATTACK_THROTTLE_NORM * 2.0
                            out_pitch = self.ATTACK_PITCH_STICK

                        # Pitch→throttle feed-forward: компенсація просідання при зміні нахилу
                        # thr_ff = self._pitch_throttle_feedforward(out_pitch)
                        ly = max(-1.0, min(1.0, ly + 0)) #thr_ff))

                        self.drone.set_sticks(lx, ly, out_roll, out_pitch)

                        # Оновлюємо trim лише коли yaw неактивний — інакше
                        # засмітимо trim паразитним flow_x від повороту.
                        if not yaw_active:
                            pitch_residual = self.brake.flow_div - self.brake.forward_setpoint
                            self._update_trim(out_roll, out_pitch,
                                              self.brake.flow_x, pitch_residual)

                # elif self.flight_mode == "AUTO" and target is not None:
                #     cx_t, cy_t = target.center
                #     err_x = (cx_t - cx_screen) / (width / 2)
                #     err_y = (cy_t - cy_screen) / (height / 2)
                #     bbox_ratio = target.width / max(width, 1)
                #     lx, ly, rx, ry = self.autopilot.compute(err_x, err_y, bbox_ratio)
                #     self.drone.set_sticks(lx, ly, rx, ry)

                # elif self.flight_mode == "AUTO":
                #     lx, ly, rx, ry = self.autopilot.get_hold_commands()
                #     self.drone.set_sticks(lx, ly, rx, ry)

                elif self.flight_mode in ("MANUAL", "LOCK"):
                    p_roll, p_pitch, p_throttle, p_yaw = self.joystick.read()
                    self.drone.set_sticks(p_yaw, p_throttle, p_roll, p_pitch)

                # ── Відображення у вікні ──
                display_frame = current_frame.copy()
                visualizer.draw_crosshair(display_frame, cx_screen, cy_screen)
                # visualizer.draw_detections(
                #     display_frame, detections,
                #     self.locked_track_id if self.flight_mode != "MANUAL" else None,
                # )

                # Режим + FPS
                fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
                cv2.putText(display_frame, f"CAM: {self._cam_fps:.1f} FPS",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                mode_text = f"{self.flight_mode}"
                # if self.flight_mode == "AUTO":
                #     mode_text += f" | {self.autopilot.attack_phase}"
                # cv2.putText(display_frame, f"[{mode_text}] FPS: {infer_fps:.0f} objs: {len(detections)}",
                #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if self.flight_mode == "BRAKE":
                    self._draw_brake_overlay(display_frame, cx_screen, cy_screen)
                    cv2.putText(display_frame,
                                f"flow X={self.brake.flow_x:+.2f} Y={self.brake.flow_y:+.2f} div={self.brake.flow_div:+.2f}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(display_frame,
                                f"trim R={self.trim_roll:+.3f} P={self.trim_pitch:+.3f} (n={len(self._trim_history)})",
                                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                    cv2.putText(display_frame,
                                f"fwd set={self.brake.forward_setpoint:+.2f} [w/s/x]",
                                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)
                    yaw_cmd_display = self._current_yaw_cmd()
                    yaw_color = (0, 200, 255) if self._is_yaw_active() else (120, 120, 120)
                    cv2.putText(display_frame,
                                f"yaw={yaw_cmd_display:+.2f} [a/d] {'ACTIVE' if self._is_yaw_active() else 'idle'}",
                                (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yaw_color, 2)

                    # HUNT HUD
                    if self.hunt_active and self.hunt is not None:
                        phase = self.hunt.phase
                        phase_color = {
                            "SCAN_ROTATE":  (0, 200, 255),
                            "SCAN_SETTLE":  (0, 255, 255),
                            "ALIGN_PULSE":  (255, 180, 0),
                            "ALIGN_SETTLE": (255, 220, 0),
                            "ATTACK":       (0, 0, 255),
                        }.get(phase, (200, 200, 200))
                        cv2.putText(display_frame,
                                    f"HUNT: {phase}  id={self.hunt.locked_track_id}",
                                    (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color, 2)
                        if self.hunt.target is not None:
                            t = self.hunt.target
                            cv2.rectangle(display_frame, (t.x1, t.y1), (t.x2, t.y2), phase_color, 2)
                            tcx = (t.x1 + t.x2) // 2
                            tcy = (t.y1 + t.y2) // 2
                            cv2.line(display_frame, (cx_screen, cy_screen), (tcx, tcy), phase_color, 2)
                            cv2.putText(display_frame, f"conf={t.confidence:.2f}",
                                        (t.x1, max(20, t.y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color, 1)

                # Телеметрія гіроскопа
                if telem_pitch is not None:
                    cv2.putText(display_frame,
                                f"GYRO P:{telem_pitch:+.1f} R:{telem_roll:+.1f}",
                                (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                # if target is not None and self.flight_mode in ("LOCK", "AUTO"):
                #     cx_t, cy_t = target.center
                #     cv2.line(display_frame, (cx_screen, cy_screen), (cx_t, cy_t), (255, 255, 0), 2)
                #     cv2.putText(display_frame,
                #                 f"ID={target.track_id} dx={cx_t - cx_screen:+d} dy={cy_t - cy_screen:+d}",
                #                 (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                cv2.imshow(self.WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                # Консольний вивід
                if target is not None:
                    cx_t, cy_t = target.center
                    print(f"\r[{self.flight_mode}] ID={target.track_id} "
                          f"dx={cx_t - cx_screen:+5d} dy={cy_t - cy_screen:+5d} "
                          f"objs={len(detections)} FPS={infer_fps:.0f}   ", end="")
                else:
                    print(f"\r[{self.flight_mode}] no target | "
                          f"objs={len(detections)} FPS={infer_fps:.0f}   ", end="")

        finally:
            self._infer_running = False
            infer_thread.join(timeout=2)
            if self.hunt is not None:
                self.hunt.shutdown()
            self.drone.reset()
            self.drone.close()
            keyboard.unhook_all()
            self.joystick.quit()
            self.source.stop()
            cv2.destroyAllWindows()

        print("\n[DRONE] Завершено.")
