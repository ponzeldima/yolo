"""Головний конвеєр (Pipeline): оркеструє всі компоненти системи."""

import time
import threading

import numpy as np
import keyboard

from . import config
from .models import Detection
from .image_source import BaseImageSource
from .tracker import BaseTracker
from .controller import PhysicalJoystick, VirtualGamepad
from .autopilot import AutoPilot
from .brake import OpticalFlowBrake
from .display import BaseDisplay
from . import visualizer


class DroneAimPipeline:
    """Composition root: з'єднує image source, tracker і display у єдиний цикл."""

    def __init__(self, image_source: BaseImageSource, tracker: BaseTracker, display: BaseDisplay):
        self.source = image_source
        self.tracker = tracker
        self.display = display

        self.joystick = PhysicalJoystick()
        self.gamepad = VirtualGamepad()
        self.autopilot = AutoPilot()
        self.brake = OpticalFlowBrake()

        # ── Стан ──
        self.flight_mode = "MANUAL"  # MANUAL | LOCK | AUTO | BRAKE
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
        self.gamepad.reset()  # скидаємо віртуальний геймпад в нейтраль
        print("\n[MODE] MANUAL — фізичний контролер напряму")

    def _toggle_mode(self, _event=None):
        if self.flight_mode == "BRAKE":
            return  # в BRAKE-режимі SPACE ігнорується, вихід тільки через B
        if self.flight_mode == "MANUAL":
            if self.closest_track_id is not None:
                self.flight_mode = "LOCK"
                self.locked_track_id = self.closest_track_id
                print(f"\n[MODE] LOCK — ціль ID={self.locked_track_id} залоковано, керування з джойстіка")
            else:
                print("\n[НЕМАЄ ЦІЛІ] Немає машини для локу")
        elif self.flight_mode == "LOCK":
            self.flight_mode = "AUTO"
            self.autopilot.reset()
            self.autopilot.attack_phase = "APPROACH"
            print(f"\n[MODE] AUTO — атака на ID={self.locked_track_id}!")
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
            print("\n[MODE] BRAKE — optical flow гальмування")

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
            self._infer_fps = 1.0 / (dt + 1e-9)
            with self._infer_lock:
                # self._latest_detections = detections
                self._latest_frame = frame
                self._infer_seq += 1

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

    # ── Головний цикл ────────────────────────────────────────────────────────

    def run(self):
        # Ініціалізація внутрішніх компонентів
        self.joystick.init()
        self.gamepad.init()

        left, top, width, height = self.source.get_region()
        cx_screen = width // 2
        cy_screen = height // 2

        # Клавіатура
        keyboard.on_press_key("space", self._toggle_mode)
        keyboard.on_press_key("b", self._toggle_brake)
        keyboard.on_press_key("q", lambda _: setattr(self, '_quit_flag', True))

        # Запуск фонового інференсу
        self._infer_running = True
        infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        infer_thread.start()

        print(f"[INFO] Область: {width}x{height}")
        print("[INFO] ПРОБІЛ — MANUAL → LOCK → AUTO → MANUAL")
        print("[INFO] 'b' — BRAKE (optical flow зависання)")
        print("[INFO] 'q' — вихід")
        print("[INFO] В Uncrashed прив'яжи керування до Xbox 360 Controller!\n")
        print("[MODE] MANUAL — passthrough фізичного контролера\n")

        # Профілювання
        _prof = {k: 0.0 for k in ('cap', 'logic', 'draw', 'overlay', 'total')}
        _prof_n = 0
        frame_count = 0

        try:
            while not self._quit_flag:
                t0 = time.perf_counter()
                frame_count += 1

                # Оновлення позиції вікна (кожні ~600 кадрів)
                if frame_count % 600 == 0 and hasattr(self.source, 'refresh_window'):
                    if self.source.refresh_window():
                        left, top, width, height = self.source.get_region()
                        cx_screen, cy_screen = width // 2, height // 2
                        self.display.reposition(left, top, width, height)

                # Отримання останніх детекцій
                t_cap = time.perf_counter()
                with self._infer_lock:
                    detections = list(self._latest_detections)
                    seq = self._infer_seq
                    current_frame = self._latest_frame
                t_cap = time.perf_counter() - t_cap

                if seq == 0:
                    time.sleep(0.001)
                    continue  # ще немає першого результату інференсу

                # ── Вибір цілі ──
                t_logic = time.perf_counter()
                target = None

                if self.flight_mode == "AUTO":
                    if self.locked_track_id is not None:
                        target = self._find_by_track_id(detections, self.locked_track_id)
                        if target:
                            self.lost_since = None
                        else:
                            if self.lost_since is None:
                                self.lost_since = time.perf_counter()
                            if (time.perf_counter() - self.lost_since) >= config.LOST_TIMEOUT:
                                print(f"\n[ВТРАТА] Ціль ID={self.locked_track_id} зникла на {config.LOST_TIMEOUT}с → MANUAL")
                                self._switch_to_manual()

                elif self.flight_mode == "LOCK":
                    if self.locked_track_id is not None:
                        target = self._find_by_track_id(detections, self.locked_track_id)
                        if target:
                            self.lost_since = None
                        else:
                            if self.lost_since is None:
                                self.lost_since = time.perf_counter()
                            if (time.perf_counter() - self.lost_since) >= config.LOST_TIMEOUT:
                                print(f"\n[ВТРАТА] Ціль ID={self.locked_track_id} зникла → MANUAL")
                                self._switch_to_manual()

                else:  # MANUAL
                    closest = self._pick_closest_to_center(detections, cx_screen, cy_screen)
                    if closest:
                        self.closest_track_id = closest.track_id
                        target = closest
                    else:
                        self.closest_track_id = None

                # ── Керування геймпадом ──
                if self.flight_mode == "BRAKE" and current_frame is not None:
                    lx, ly, rx, ry = self.brake.compute(current_frame)
                    print(f"\r[BRAKE] yaw={lx:+.3f} thr={ly:+.3f} roll={rx:+.3f} pitch={ry:+.3f}", end="")
                    self.gamepad.set_sticks(lx, ly, rx, ry)

                elif self.flight_mode == "AUTO" and target is not None:
                    cx_t, cy_t = target.center
                    err_x = (cx_t - cx_screen) / (width / 2)
                    err_y = (cy_t - cy_screen) / (height / 2)
                    bbox_ratio = target.width / max(width, 1)
                    lx, ly, rx, ry = self.autopilot.compute(err_x, err_y, bbox_ratio)
                    self.gamepad.set_sticks(lx, ly, rx, ry)

                elif self.flight_mode == "AUTO":
                    lx, ly, rx, ry = self.autopilot.get_hold_commands()
                    self.gamepad.set_sticks(lx, ly, rx, ry)

                elif self.flight_mode in ("MANUAL", "LOCK"):
                    # MANUAL / LOCK — passthrough фізичного джойстіка через віртуальний геймпад
                    p_roll, p_pitch, p_throttle, p_yaw = self.joystick.read()
                    self.gamepad.set_sticks(p_yaw, p_throttle, p_roll, p_pitch)

                t_logic = time.perf_counter() - t_logic

                # ── Малюємо HUD ──
                t_draw = time.perf_counter()
                overlay_img = visualizer.create_overlay(width, height)
                visualizer.draw_crosshair(overlay_img, cx_screen, cy_screen)
                visualizer.draw_mode(overlay_img, self.flight_mode)
                visualizer.draw_detections(
                    overlay_img, detections,
                    self.locked_track_id if self.flight_mode != "MANUAL" else None,
                )

                if self.flight_mode == "BRAKE":
                    visualizer.draw_brake_info(overlay_img, self.brake.flow_x, self.brake.flow_y,
                                              self.brake.flow_div)
                    fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
                    print(f"[BRAKE] flow_x={self.brake.flow_x:+.3f} "
                          f"flow_y={self.brake.flow_y:+.3f} | "
                          f"FPS: {fps:.1f}   ", end="\r")

                elif target is not None:
                    cx_t, cy_t = target.center
                    delta_x = cx_t - cx_screen
                    delta_y = cy_t - cy_screen
                    bbox_ratio_draw = target.width / max(width, 1)

                    if self.flight_mode == "LOCK":
                        visualizer.draw_lock_info(overlay_img, target, cx_screen, cy_screen,
                                                  self.locked_track_id)
                    elif self.flight_mode == "AUTO":
                        visualizer.draw_auto_info(overlay_img, target, cx_screen, cy_screen,
                                                  self.locked_track_id,
                                                  self.autopilot.attack_phase, width)

                    fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
                    print(f"[{self.flight_mode}|{self.autopilot.attack_phase:8s}] "
                          f"dx={delta_x:+5d} dy={delta_y:+5d} "
                          f"bbox={bbox_ratio_draw:.0%} | cars:{len(detections)} | "
                          f"FPS: {fps:.1f}   ", end="\r")
                else:
                    fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
                    phase_str = self.autopilot.attack_phase if self.flight_mode == "AUTO" else ""
                    print(f"[{self.flight_mode}] NO TARGET {phase_str} | "
                          f"FPS: {fps:.1f}   ", end="\r")

                t_draw = time.perf_counter() - t_draw

                # ── Відображення ──
                t_ovr = time.perf_counter()
                self.display.show(overlay_img)
                if self.display.should_quit():
                    break
                t_ovr = time.perf_counter() - t_ovr

                # ── Профілювання ──
                _prof_n += 1
                a = 0.05
                _prof['cap'] = _prof['cap'] * (1 - a) + t_cap * a
                _prof['logic'] = _prof['logic'] * (1 - a) + t_logic * a
                _prof['draw'] = _prof['draw'] * (1 - a) + t_draw * a
                _prof['overlay'] = _prof['overlay'] * (1 - a) + t_ovr * a
                _prof['total'] = _prof['total'] * (1 - a) + (time.perf_counter() - t0) * a

                if _prof_n % 60 == 0:
                    loop_fps = 1.0 / (_prof['total'] + 1e-9)
                    print(f"\n[PROF] cap={_prof['cap']*1000:.1f}ms  "
                          f"logic={_prof['logic']*1000:.1f}ms  "
                          f"draw={_prof['draw']*1000:.1f}ms  "
                          f"overlay={_prof['overlay']*1000:.1f}ms  "
                          f"TOTAL={_prof['total']*1000:.1f}ms  "
                          f"loopFPS={loop_fps:.0f}  inferFPS={self._infer_fps:.0f}")

        finally:
            self._infer_running = False
            infer_thread.join(timeout=2)
            self.gamepad.reset()
            keyboard.unhook_all()
            self.joystick.quit()
            self.display.destroy()
            self.source.stop()

        print("\n[INFO] Завершено.")
