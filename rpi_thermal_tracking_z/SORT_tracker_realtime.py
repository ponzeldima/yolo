#!/usr/bin/env python3
"""
MLX90642 Real-Time Thermal Tracker
===================================
Об'єднаний код: зчитування з MLX90642 (smbus2) + MRF/GC сегментація + SORT трекінг.

Сама логіка сегментації та SORT-трекінгу винесена в `trackers/` (щоб її
можна було перевикористати у скриптах прогону записаних сесій).

Залежності:
    pip install smbus2 numpy opencv-python scipy PyMaxflow

Запуск:
    python SORT_tracker_realtime.py

Клавіші під час роботи:
    Q / ESC — вихід
    S       — зберегти поточний кадр як PNG
    R       — скинути трекер (нові ID)
    +/-     — збільшити / зменшити поріг K_SIGMA
"""

import time
import os
import sys
import argparse
import numpy as np
import cv2
from smbus2 import SMBus

# ── MJPEG + thermal_viz + спільні трекери ───────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mjpeg_server import start_mjpeg_server, lan_ip_hint  # noqa: E402
from mlx90642_io import open_bus, read_frame, FRAME_H, FRAME_W  # noqa: E402
from thermal_viz import colorize  # noqa: E402
from trackers.segmentation import segment_frame, DEFAULT_LAMBDA, DEFAULT_K_SIGMA  # noqa: E402
from trackers.sort import SORTTrack, SORTTracker  # noqa: E402

# ═══════════════════════════════════════════════
#  КОНФІГУРАЦІЯ
# ═══════════════════════════════════════════════

I2C_BUS       = 1
DISP_W, DISP_H = 640, 480
FPS_TARGET    = 8

LAMBDA  = DEFAULT_LAMBDA     # MRF smoothness
K_SIGMA = DEFAULT_K_SIGMA    # mean + K_SIGMA * std

# SORT
IOU_THRESHOLD = 0.07
MAX_AGE       = 7
MIN_HITS      = 3

SAVE_DIR    = "tracker_snapshots"
WINDOW_NAME = "Thermal Tracker — Live"

SCALE_X = DISP_W / FRAME_W
SCALE_Y = DISP_H / FRAME_H


# ═══════════════════════════════════════════════
#  Сенсор / візуалізація
# ═══════════════════════════════════════════════

def read_sensor_frame(bus: SMBus) -> np.ndarray:
    return read_frame(bus)


def frame_to_bgr(frame2d: np.ndarray) -> np.ndarray:
    t_min = float(frame2d.min())
    t_max = float(frame2d.max())
    return colorize(frame2d, t_min, t_max, DISP_W, DISP_H,
                    interp=cv2.INTER_CUBIC)


# ═══════════════════════════════════════════════
#  Рендеринг
# ═══════════════════════════════════════════════

TRACK_COLORS = [
    (0, 220, 220), (220, 0, 220), (220, 220, 0),
    (0, 150, 255), (255, 100, 0), (100, 255, 0),
    (0, 255, 150), (150, 0, 255),
]


def _to_disp(r, c):
    return int(c * SCALE_X), int(r * SCALE_Y)


def render(bgr, frame2d, mask, bbox_mrf, sort_mgr,
           frame_idx, fps, trajectory, k_sigma):
    if mask is not None:
        mask_big = cv2.resize(mask.astype(np.uint8) * 255,
                              (DISP_W, DISP_H), interpolation=cv2.INTER_NEAREST)
        ov = bgr.copy()
        ov[mask_big > 0] = (
            ov[mask_big > 0] * 0.55 + np.array([180, 120, 0]) * 0.45
        ).astype(np.uint8)
        bgr[:] = ov

    if bbox_mrf is not None:
        x0, y0 = _to_disp(bbox_mrf[0], bbox_mrf[2])
        x1, y1 = _to_disp(bbox_mrf[1], bbox_mrf[3])
        cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 220, 0), 2)
        cv2.putText(bgr, "MRF/GC", (x0, max(y0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)

    for t in sort_mgr.tracks:
        r0, r1, c0, c1 = t.get_bbox()
        x0, y0 = _to_disp(r0, c0); x1, y1 = _to_disp(r1, c1)
        x0, x1 = max(0, x0), min(DISP_W - 1, x1)
        y0, y1 = max(0, y0), min(DISP_H - 1, y1)
        col   = TRACK_COLORS[(t.track_id - 1) % len(TRACK_COLORS)]
        thick = 2 if t.confirmed else 1
        cv2.rectangle(bgr, (x0, y0), (x1, y1), col, thick)
        cv2.putText(bgr, f"ID{t.track_id} h{t.hits}",
                    (x0, max(y0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        vx, vy = t.get_velocity()
        cx_d, cy_d = (x0 + x1) // 2, (y0 + y1) // 2
        vxd, vyd   = int(vx * SCALE_X * 3), int(vy * SCALE_Y * 3)
        if abs(vxd) + abs(vyd) > 2:
            cv2.arrowedLine(bgr, (cx_d, cy_d),
                            (cx_d + vxd, cy_d + vyd),
                            col, 2, tipLength=0.3)

    for j in range(1, len(trajectory)):
        alpha = j / len(trajectory)
        cv2.line(bgr, trajectory[j - 1], trajectory[j],
                 (int(255 * alpha), int(200 * alpha), 0), 1)

    best    = sort_mgr.get_best()
    best_id = best.track_id if best else -1
    vx_v, vy_v = best.get_velocity() if best else (0., 0.)

    lines = [
        f"Frame: {frame_idx:05d}  FPS: {fps:.1f}",
        f"T_min: {frame2d.min():.1f}C  T_max: {frame2d.max():.1f}C",
        f"Tracks: {len(sort_mgr.tracks)} total / {len(sort_mgr.get_active())} confirmed",
        f"Best ID: {best_id}  Vel: ({vx_v:.2f}, {vy_v:.2f})",
        f"K_sigma: {k_sigma:.1f} [+/-]  R-reset  S-save  Q-quit",
    ]
    for i, line in enumerate(lines):
        yp = 20 + i * 22
        cv2.putText(bgr, line, (8, yp), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(bgr, line, (8, yp), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (255, 255, 255), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════
#  ГОЛОВНИЙ ЦИКЛ
# ═══════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser("MLX90642 SORT tracker (live)")
    p.add_argument("--stream", action="store_true",
                   help="публікувати MJPEG на http://host:port/")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--save-dir", default=None,
                   help="зберігати PNG-кадри в цю теку (headless)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="вийти після N кадрів (0 = без обмеження)")
    p.add_argument("--jpeg-quality", type=int, default=80)
    return p.parse_args()


class _Display:
    """Об'єднує MJPEG-стрім / збереження PNG / локальне вікно."""

    def __init__(self, args):
        self.headless = bool(args.stream or args.save_dir)
        self.save_dir = args.save_dir
        self.bus = None
        self.httpd = None
        self.jpeg_quality = args.jpeg_quality
        if args.stream:
            self.bus, self.httpd = start_mjpeg_server(args.host, args.port)
            ip = lan_ip_hint() if args.host in ("0.0.0.0", "") else args.host
            print(f"MJPEG stream: http://{ip}:{args.port}/")
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            print(f"Saving frames to: {args.save_dir}")
        if not self.headless:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, DISP_W, DISP_H)

    def publish(self, bgr, frame_idx):
        if self.bus is not None:
            ok, jpg = cv2.imencode(".jpg", bgr,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                self.bus.publish(jpg.tobytes())
        if self.save_dir:
            cv2.imwrite(os.path.join(self.save_dir,
                                     f"frame_{frame_idx:05d}.png"), bgr)
        if not self.headless:
            cv2.imshow(WINDOW_NAME, bgr)

    def wait_key(self, ms: int) -> int:
        if self.headless:
            time.sleep(max(ms, 1) / 1000.0)
            return 0xFF
        return cv2.waitKey(ms) & 0xFF

    def close(self):
        if self.bus is not None:
            self.bus.stop()
        if self.httpd is not None:
            self.httpd.shutdown()
        if not self.headless:
            cv2.destroyAllWindows()


def main():
    args = _parse_args()
    os.makedirs(SAVE_DIR, exist_ok=True)
    SORTTrack._id_counter = 0
    sort = SORTTracker(iou_threshold=IOU_THRESHOLD,
                       max_age=MAX_AGE,
                       min_hits=MIN_HITS)
    trajectory = []
    frame_idx  = 0
    fps        = 0.0
    k_sigma    = K_SIGMA

    delay = 1.0 / FPS_TARGET

    display = _Display(args)

    print("Підключення до MLX90642...")
    try:
        bus = open_bus()
    except Exception as e:
        print(f"Помилка відкриття I2C-шини {I2C_BUS}: {e}")
        return

    time.sleep(0.5)
    print("Сенсор готовий. Запуск трекінгу... (Q/ESC — вихід)")

    t_fps  = time.perf_counter()
    fps_n  = 0

    try:
        while True:
            t0 = time.perf_counter()

            try:
                frame2d = read_sensor_frame(bus)
            except Exception as e:
                print(f"[{frame_idx}] Помилка читання сенсора: {e}")
                time.sleep(0.1)
                continue

            mask, bbox_mrf = segment_frame(frame2d, k_sigma=k_sigma, lam=LAMBDA)
            detections = [bbox_mrf] if bbox_mrf is not None else []

            sort.predict_all()
            sort.update(detections)

            best = sort.get_best()
            if best is not None:
                r0k, r1k, c0k, c1k = best.get_bbox()
                cx_d = int((c0k + c1k) / 2 * SCALE_X)
                cy_d = int((r0k + r1k) / 2 * SCALE_Y)
                trajectory.append((cx_d, cy_d))
                if len(trajectory) > 60:
                    trajectory.pop(0)

            bgr = frame_to_bgr(frame2d)
            render(bgr, frame2d, mask, bbox_mrf, sort,
                   frame_idx, fps, trajectory, k_sigma)
            display.publish(bgr, frame_idx)

            fps_n += 1
            if fps_n >= 10:
                fps   = fps_n / (time.perf_counter() - t_fps + 1e-9)
                t_fps = time.perf_counter()
                fps_n = 0

            elapsed  = time.perf_counter() - t0
            wait_ms  = max(1, int((delay - elapsed) * 1000))
            key = display.wait_key(wait_ms)

            if not display.headless:
                if key in (ord('q'), 27):
                    break
                elif key == ord('s'):
                    fname = os.path.join(SAVE_DIR, f"frame_{frame_idx:05d}.png")
                    cv2.imwrite(fname, bgr)
                    print(f"Збережено: {fname}")
                elif key == ord('r'):
                    sort.reset()
                    trajectory.clear()
                    print("Трекер скинуто.")
                elif key == ord('+'):
                    k_sigma = min(k_sigma + 0.5, 8.0)
                    print(f"K_sigma = {k_sigma:.1f}")
                elif key == ord('-'):
                    k_sigma = max(k_sigma - 0.5, 0.5)
                    print(f"K_sigma = {k_sigma:.1f}")

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break

    except KeyboardInterrupt:
        print("\nПерервано користувачем.")
    finally:
        bus.close()
        display.close()
        print(f"Завершено. Оброблено {frame_idx} кадрів.")


if __name__ == "__main__":
    main()
