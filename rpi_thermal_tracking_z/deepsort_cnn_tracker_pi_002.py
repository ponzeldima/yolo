#!/usr/bin/env python3
"""
MLX90642 DeepSORT + MicroReID-IR  —  Raspberry Pi 3B+
=====================================================
Логіка DeepSORT з CNN-аппірансом винесена в trackers.deepsort_cnn,
сегментація — у trackers.segmentation.

Залежності: numpy scipy PyMaxflow opencv-python-headless torch
"""

import time
import csv
import os
import sys
import argparse
import numpy as np
import cv2

# ── MJPEG (+ опційно сенсор) + thermal_viz + спільні трекери ────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mjpeg_server import start_mjpeg_server, lan_ip_hint  # noqa: E402
from thermal_viz import colorize  # noqa: E402
from trackers.segmentation import (  # noqa: E402
    segment_frame, FRAME_H, FRAME_W,
    DEFAULT_LAMBDA, DEFAULT_K_SIGMA,
)
from trackers.deepsort_cnn import (  # noqa: E402
    DeepSORTCNNTrack as DeepSORTTrack,
    DeepSORTCNNTracker as DeepSORTManager,
    load_model,
    DEFAULT_MAX_AGE,
    DEFAULT_MIN_HITS,
    DEFAULT_LAMBDA_COST,
    DEFAULT_GALLERY_SIZE,
    DEFAULT_GATE_CHI2,
    DEFAULT_IOU_THR,
    DEFAULT_PATCH_H,
    DEFAULT_PATCH_W,
    DEFAULT_EMBED_DIM,
)

# ── CONFIG ────────────────────────────────────────────────
CSV_PATH      = "measurement_01_MD.csv"
MODEL_PATH    = "micro_reid_ir.pt"
FPS           = 10
DISP_W        = 640
DISP_H        = 480
PATCH_H       = DEFAULT_PATCH_H
PATCH_W       = DEFAULT_PATCH_W
EMBED_DIM     = DEFAULT_EMBED_DIM

LAMBDA        = DEFAULT_LAMBDA
K_SIGMA       = DEFAULT_K_SIGMA

MAX_AGE       = DEFAULT_MAX_AGE
MIN_HITS      = DEFAULT_MIN_HITS
LAMBDA_COST   = DEFAULT_LAMBDA_COST
GALLERY_SIZE  = DEFAULT_GALLERY_SIZE
GATE_CHI2     = DEFAULT_GATE_CHI2
IOU_THRESHOLD = DEFAULT_IOU_THR

DEVICE        = "cpu"
WINDOW_NAME   = "Thermal DeepSORT CNN"

# Шкала температур (оновлюється з CSV / live)
T_MIN: float = 20.0
T_MAX: float = 60.0
FEAT_DIM: int = EMBED_DIM
# ─────────────────────────────────────────────────────────


def load_csv(path):
    rows, timestamps = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f, delimiter=";"):
            if len(row) < 2:
                continue
            try:
                t = np.array(row[1:], dtype=np.float32)
            except ValueError:
                continue
            if t.size == FRAME_H * FRAME_W:
                timestamps.append(row[0])
                rows.append(t)
    if not rows:
        raise RuntimeError(f"Немає даних у {path}")
    print(f"Завантажено {len(rows)} кадрів")
    return np.stack(rows), timestamps


# ── Display: MJPEG / save PNG / cv2 window ────────────────
class _Display:
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

    def publish(self, bgr, idx):
        if self.bus is not None:
            ok, jpg = cv2.imencode(".jpg", bgr,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                self.bus.publish(jpg.tobytes())
        if self.save_dir:
            cv2.imwrite(os.path.join(self.save_dir, f"frame_{idx:05d}.png"), bgr)
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


def _parse_args():
    p = argparse.ArgumentParser("DeepSORT CNN ReID (MicroReIDIR)")
    p.add_argument("--source", choices=["csv", "sensor"], default="csv")
    p.add_argument("--csv", default=CSV_PATH)
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--jpeg-quality", type=int, default=80)
    return p.parse_args()


# ── Рендеринг ─────────────────────────────────────────────
_COLORS = [(0, 220, 220), (220, 0, 220), (220, 220, 0),
           (0, 150, 255), (255, 100, 0), (100, 255, 0)]


def render_frame_deepsort(frame_2d, engine_mask, bbox_mrf, manager,
                          frame_idx, n_frames, timestamp, trajectory, fps):
    scale_x = DISP_W / FRAME_W
    scale_y = DISP_H / FRAME_H

    bgr = colorize(frame_2d, T_MIN, T_MAX, DISP_W, DISP_H)

    if engine_mask is not None:
        mask_big = cv2.resize(engine_mask.astype(np.uint8) * 255,
                              (DISP_W, DISP_H), interpolation=cv2.INTER_NEAREST)
        ov = bgr.copy()
        ov[mask_big > 0] = (ov[mask_big > 0] * 0.55 +
                            np.array([180, 120, 0]) * 0.45).astype(np.uint8)
        bgr = ov

    def tod(r, c): return int(c * scale_x), int(r * scale_y)

    if bbox_mrf is not None:
        r0, r1, c0, c1 = bbox_mrf
        cv2.rectangle(bgr, tod(r0, c0), tod(r1, c1), (0, 220, 0), 2)
        cv2.putText(bgr, "MRF+GC", (tod(r0, c0)[0], max(tod(r0, c0)[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)

    for track in manager.tracks:
        r0, r1, c0, c1 = track.get_bbox()
        x0, y0 = tod(r0, c0); x1, y1 = tod(r1, c1)
        x0, x1 = max(0, x0), min(DISP_W - 1, x1)
        y0, y1 = max(0, y0), min(DISP_H - 1, y1)
        clr = _COLORS[(track.track_id - 1) % len(_COLORS)]
        cv2.rectangle(bgr, (x0, y0), (x1, y1), clr, 2 if track.confirmed else 1)
        cv2.putText(bgr, f"ID{track.track_id} h{track.hits} g{len(track.gallery)}",
                    (x0, max(y0 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, clr, 1, cv2.LINE_AA)
        vx, vy = track.get_velocity()
        cxd, cyd = int((x0 + x1) / 2), int((y0 + y1) / 2)
        vxd, vyd = int(vx * scale_x * 3), int(vy * scale_y * 3)
        if abs(vxd) + abs(vyd) > 2:
            cv2.arrowedLine(bgr, (cxd, cyd), (cxd + vxd, cyd + vyd),
                            clr, 2, tipLength=0.3)

    for j in range(1, len(trajectory)):
        a = j / len(trajectory)
        cv2.line(bgr, trajectory[j - 1], trajectory[j],
                 (int(255 * a), int(200 * a), 0), 1)

    best = manager.get_best()
    t_max_frame   = frame_2d.max()
    t_max_engine  = frame_2d[engine_mask].max()  if engine_mask is not None else 0.
    t_mean_engine = frame_2d[engine_mask].mean() if engine_mask is not None else 0.
    t_mean_bg     = frame_2d[~engine_mask].mean() if engine_mask is not None else frame_2d.mean()
    vxv, vyv = best.get_velocity() if best else (0., 0.)

    lines = [
        f"Frame {frame_idx:03d}/{n_frames - 1}  {timestamp}  FPS:{fps:.1f}",
        f"T frame max: {t_max_frame:.1f}C",
        f"T engine max: {t_max_engine:.1f}C | avg: {t_mean_engine:.1f}C",
        f"T background: {t_mean_bg:.1f}C",
        f"DeepSORT: {len(manager.tracks)} tracks / {len(manager.get_active())} confirmed",
        f"lam={LAMBDA_COST} gallery={GALLERY_SIZE} gate={GATE_CHI2:.2f}  CNN dim={FEAT_DIM}",
        f"Best ID: {best.track_id if best else -1}  Vel: ({vxv:.2f},{vyv:.2f})",
    ]
    for i, line in enumerate(lines):
        yp = 20 + i * 22
        cv2.putText(bgr, line, (8, yp), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(bgr, line, (8, yp), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


# ── Головний цикл ─────────────────────────────────────────
def run_deepsort_tracker(args=None):
    global FEAT_DIM, T_MIN, T_MAX
    if args is None:
        args = _parse_args()
    DeepSORTTrack._id_counter = 0

    reid_model, FEAT_DIM = load_model(args.model, device=DEVICE)
    print(f"Модель завантажена: {args.model}  embed_dim={FEAT_DIM}")

    sensor_bus = None
    temps = None
    timestamps: list[str] = []

    if args.source == "csv":
        temps, timestamps = load_csv(args.csv)
        n_frames = len(temps)
        T_MIN, T_MAX = float(temps.min()), float(temps.max())
    else:
        from mlx90642_io import open_bus, read_frame  # noqa: WPS433
        sensor_bus = open_bus()
        n_frames   = int(1e9)
        T_MIN, T_MAX = 20., 60.

    manager = DeepSORTManager(reid_model, FEAT_DIM, T_MIN, T_MAX,
                              max_age=MAX_AGE,
                              min_hits=MIN_HITS,
                              lambda_cost=LAMBDA_COST,
                              gallery_size=GALLERY_SIZE,
                              gate_chi2=GATE_CHI2,
                              iou_thr=IOU_THRESHOLD,
                              patch_h=PATCH_H,
                              patch_w=PATCH_W,
                              device=DEVICE)
    trajectory = []

    display = _Display(args)

    delay    = 1.0 / FPS
    t_start  = time.perf_counter()
    fps_disp = 0.0
    print(f"DeepSORT CNN | source={args.source} | Q/Esc для виходу")

    frame_idx = -1
    try:
        for frame_idx in range(n_frames):
            t0 = time.perf_counter()
            if args.source == "csv":
                flat  = temps[frame_idx]
                frame = flat.reshape(FRAME_H, FRAME_W)
                ts    = timestamps[frame_idx] if timestamps else ""
            else:
                frame = read_frame(sensor_bus)
                T_MIN = 0.9 * T_MIN + 0.1 * float(frame.min())
                T_MAX = 0.9 * T_MAX + 0.1 * float(frame.max())
                manager.set_temp_range(T_MIN, T_MAX)
                ts    = ""

            engine_mask, bbox_mrf = segment_frame(frame, k_sigma=K_SIGMA, lam=LAMBDA)
            detections = [bbox_mrf] if bbox_mrf is not None else []

            manager.predict_all()
            manager.update(detections, frame)

            best = manager.get_best()
            if best is not None:
                r0k, r1k, c0k, c1k = best.get_bbox()
                trajectory.append((int((c0k + c1k) / 2 * DISP_W / FRAME_W),
                                   int((r0k + r1k) / 2 * DISP_H / FRAME_H)))
                if len(trajectory) > 60:
                    trajectory.pop(0)

            bgr = render_frame_deepsort(frame, engine_mask, bbox_mrf, manager,
                                        frame_idx, n_frames, ts, trajectory, fps_disp)
            display.publish(bgr, frame_idx)

            elapsed = time.perf_counter() - t0
            wait_ms = max(1, int((delay - elapsed) * 1000))
            key = display.wait_key(wait_ms)
            if not display.headless and key in (ord("q"), 27):
                break

            if frame_idx % 10 == 9:
                fps_disp = 10 / (time.perf_counter() - t_start + 1e-9)
                t_start  = time.perf_counter()

            if args.max_frames and frame_idx + 1 >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        display.close()
        if sensor_bus is not None:
            sensor_bus.close()

    total = frame_idx + 1
    print(f"Готово: {total} кадрів")


if __name__ == "__main__":
    run_deepsort_tracker()
