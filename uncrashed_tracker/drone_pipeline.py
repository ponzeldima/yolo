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
            self._brake_start_time = time.perf_counter()
            print("\n[MODE] BRAKE — optical flow зависання (зліт 0.5с)")

    # ── Фоновий інференс ─────────────────────────────────────────────────────

    def _inference_loop(self):
        while self._infer_running:
            frame = self.source.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            t = time.perf_counter()
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
        keyboard.on_press_key("q", lambda _: setattr(self, '_quit_flag', True))

        # Фоновий інференс
        self._infer_running = True
        infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        infer_thread.start()

        print(f"[DRONE] Камера: {width}x{height}")
        print(f"[DRONE] Порт: {self.drone._port}")
        print("[DRONE] ПРОБІЛ — MANUAL → LOCK → AUTO → MANUAL")
        print("[DRONE] 'b' — BRAKE (optical flow зависання)")
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
                    time.sleep(0.01)
                    continue

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
                    if brake_elapsed < 1.1:
                        # Фаза зльоту: базовий газ, нейтраль roll/pitch
                        base_thr = -1.0 + config.BRAKE_BASE_THROTTLE * 2.0
                        self.drone.set_sticks(0.0, base_thr, 0.012, 0.07)
                    else:
                        lx, ly, rx, ry = self.brake.compute(current_frame)
                        self.drone.set_sticks(lx, ly, 0.012, 0.07)

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
            self.drone.reset()
            self.drone.close()
            keyboard.unhook_all()
            self.joystick.quit()
            self.source.stop()
            cv2.destroyAllWindows()

        print("\n[DRONE] Завершено.")
