"""Спрощений конвеєр для камери: детекція + трекінг + відображення у вікні.

Без джойстіка, автопілота і віртуального геймпада.
Використовує ті самі трекери та візуалізацію, що й симулятор.
"""

import time
import threading

import cv2
import keyboard

from .models import Detection
from .image_source import BaseImageSource
from .tracker import BaseTracker
from . import visualizer


class CameraTrackingPipeline:
    """Capture → Track → Draw → Display у вікні OpenCV (без автопілота)."""

    WINDOW_NAME = "Camera Tracker"

    def __init__(self, image_source: BaseImageSource, tracker: BaseTracker):
        self.source = image_source
        self.tracker = tracker

        self._quit_flag = False

        # ── Async inference ──
        self._infer_lock = threading.Lock()
        self._latest_detections: list[Detection] = []
        self._infer_fps = 0.0
        self._infer_seq = 0
        self._infer_running = False
        self._latest_frame = None

    # ── Фоновий інференс ─────────────────────────────────────────────────────

    def _inference_loop(self):
        while self._infer_running:
            frame = self.source.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            t = time.perf_counter()
            detections = self.tracker.track(frame)
            dt = time.perf_counter() - t
            fps = 1.0 / (dt + 1e-9)
            with self._infer_lock:
                self._latest_detections = detections
                self._infer_fps = fps
                self._infer_seq += 1
                self._latest_frame = frame.copy()

    # ── Головний цикл ────────────────────────────────────────────────────────

    def run(self):
        _, _, width, height = self.source.get_region()
        cx_screen = width // 2
        cy_screen = height // 2

        keyboard.on_press_key("q", lambda _: setattr(self, '_quit_flag', True))

        # Запуск фонового інференсу
        self._infer_running = True
        infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        infer_thread.start()

        print(f"[CAMERA] Роздільність: {width}x{height}")
        print("[CAMERA] 'q' — вихід")

        try:
            while not self._quit_flag:
                with self._infer_lock:
                    detections = list(self._latest_detections)
                    seq = self._infer_seq
                    infer_fps = self._infer_fps
                    frame = self._latest_frame

                if seq == 0 or frame is None:
                    time.sleep(0.001)
                    continue

                # Малюємо bounding boxes на реальному кадрі
                display_frame = frame.copy()
                visualizer.draw_crosshair(display_frame, cx_screen, cy_screen)
                visualizer.draw_detections(display_frame, detections)

                # FPS
                cv2.putText(display_frame, f"FPS: {infer_fps:.0f}  objs: {len(detections)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow(self.WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        finally:
            self._infer_running = False
            infer_thread.join(timeout=2)
            keyboard.unhook_all()
            self.source.stop()
            cv2.destroyAllWindows()

        print("\n[CAMERA] Завершено.")
