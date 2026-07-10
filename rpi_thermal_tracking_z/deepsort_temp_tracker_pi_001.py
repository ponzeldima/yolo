#!/usr/bin/env python3
"""
MLX90642 DeepSORT Thermal Tracker — Raspberry Pi 3B+
=====================================================
DeepSORT з температурною гістограмою як аппірансом. Логіка треккера
винесена в trackers.deepsort_temp і trackers.segmentation, тут лишається
лише ввід (CSV/сенсор), рендер і головний цикл.

pip install numpy scipy PyMaxflow opencv-python-headless
"""

import time
import csv
import os
import sys
import argparse
import numpy as np
import cv2

# ── MJPEG + сенсор + thermal_viz + спільні трекери ──────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mjpeg_server import start_mjpeg_server, lan_ip_hint  # noqa: E402
from thermal_viz import colorize  # noqa: E402
from trackers.segmentation import (  # noqa: E402
    segment_frame, FRAME_H, FRAME_W,
    DEFAULT_LAMBDA, DEFAULT_K_SIGMA,
)
from trackers.deepsort_temp import (  # noqa: E402
    DeepSORTTempTrack as Track,
    DeepSORTTempTracker as Tracker,
    set_temp_range,
    DEFAULT_MAX_AGE,
    DEFAULT_MIN_HITS,
    DEFAULT_LAMBDA_COST,
    DEFAULT_GALLERY_SIZE,
    DEFAULT_N_FEAT_BINS,
    DEFAULT_GATE_CHI2,
    DEFAULT_IOU_THR,
)

# ── CONFIG ────────────────────────────────────────────────
CSV_PATH      = "measurement_01_MD.csv"
FPS_TARGET    = 10
DISP_W, DISP_H = 640, 480

# MRF
MRF_LAMBDA = DEFAULT_LAMBDA
K_SIGMA    = DEFAULT_K_SIGMA

# DeepSORT (з модуля trackers.deepsort_temp)
MAX_AGE      = DEFAULT_MAX_AGE
MIN_HITS     = DEFAULT_MIN_HITS
LAMBDA_COST  = DEFAULT_LAMBDA_COST
GALLERY_SIZE = DEFAULT_GALLERY_SIZE
N_FEAT_BINS  = DEFAULT_N_FEAT_BINS
GATE_CHI2    = DEFAULT_GATE_CHI2
IOU_THR      = DEFAULT_IOU_THR

WINDOW_NAME = "Thermal DeepSORT"
# ─────────────────────────────────────────────────────────


def load_csv(path):
    rows, timestamps = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f, delimiter=";"):
            if len(row) < 2:
                continue
            try:
                temps = np.array(row[1:], dtype=np.float32)
            except ValueError:
                continue
            if temps.size == FRAME_H * FRAME_W:
                timestamps.append(row[0])
                rows.append(temps)
    if not rows:
        raise RuntimeError(f"No valid frames in {path}")
    print(f"Loaded {len(rows)} frames")
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
    p = argparse.ArgumentParser("DeepSORT (temperature-hist ReID)")
    p.add_argument("--source", choices=["csv", "sensor"], default="csv")
    p.add_argument("--csv", default=CSV_PATH, help="шлях до CSV")
    p.add_argument("--stream", action="store_true",
                   help="MJPEG на http://host:port/")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--jpeg-quality", type=int, default=80)
    return p.parse_args()


# ── COLORMAP (винесено в thermal_viz.colorize) ────────────
SCALE_X  = DISP_W / FRAME_W
SCALE_Y  = DISP_H / FRAME_H


def frame_to_bgr(f2d):
    """float32 (H,W) → uint8 BGR (DISP_H, DISP_W) через спільний colorize()."""
    return colorize(f2d, float(f2d.min()), float(f2d.max()),
                    DISP_W, DISP_H)


# ── Render ────────────────────────────────────────────────
_COLORS = [(0, 220, 220), (220, 0, 220), (220, 220, 0),
           (0, 150, 255), (255, 100, 0), (100, 255, 0)]


def to_d(r, c):
    return int(c * SCALE_X), int(r * SCALE_Y)


def render(bgr, f2d, mask, bbox_mrf, tracker, fidx, n_frames, fps, traj):
    if mask is not None:
        mb = cv2.resize(mask.astype(np.uint8) * 255,
                        (DISP_W, DISP_H), interpolation=cv2.INTER_NEAREST)
        ov = bgr.copy()
        ov[mb > 0] = (ov[mb > 0] * 0.55 + np.array([180, 120, 0]) * 0.45).astype(np.uint8)
        bgr[:] = ov

    if bbox_mrf is not None:
        r0, r1, c0, c1 = bbox_mrf
        cv2.rectangle(bgr, to_d(r0, c0), to_d(r1, c1), (0, 220, 0), 2)
        cv2.putText(bgr, "MRF/GC", (to_d(r0, c0)[0], max(to_d(r0, c0)[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)

    for t in tracker.tracks:
        r0, r1, c0, c1 = t.get_bbox()
        x0, y0 = to_d(r0, c0); x1, y1 = to_d(r1, c1)
        x0, x1 = max(0, x0), min(DISP_W - 1, x1)
        y0, y1 = max(0, y0), min(DISP_H - 1, y1)
        clr = _COLORS[(t.track_id - 1) % len(_COLORS)]
        cv2.rectangle(bgr, (x0, y0), (x1, y1), clr, 2 if t.confirmed else 1)
        cv2.putText(bgr, f"ID{t.track_id} h{t.hits} g{len(t.gallery)}",
                    (x0, max(y0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1, cv2.LINE_AA)
        vx, vy = t.get_velocity()
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        dx, dy = int(vx * SCALE_X * 3), int(vy * SCALE_Y * 3)
        if abs(dx) + abs(dy) > 2:
            cv2.arrowedLine(bgr, (cx, cy), (cx + dx, cy + dy), clr, 2, tipLength=0.3)

    for j in range(1, len(traj)):
        a = j / len(traj)
        cv2.line(bgr, traj[j - 1], traj[j],
                 (int(255 * a), int(200 * a), 0), 1)

    b = tracker.get_best()
    vx_v, vy_v = b.get_velocity() if b else (0., 0.)
    lines = [
        f"Frame {fidx:03d}/{n_frames - 1}  FPS:{fps:.1f}",
        f"T_max:{f2d.max():.1f}C",
        f"Tracks:{len(tracker.tracks)} conf:{len(tracker.get_active())}",
        f"Best ID:{b.track_id if b else -1}  Vel:({vx_v:.2f},{vy_v:.2f})",
        f"lam={LAMBDA_COST} gal={GALLERY_SIZE} gate={GATE_CHI2:.2f}",
    ]
    for i, line in enumerate(lines):
        yp = 20 + i * 22
        cv2.putText(bgr, line, (8, yp), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(bgr, line, (8, yp), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (255, 255, 255), 1, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────
def run(args=None):
    if args is None:
        args = _parse_args()
    Track._id_counter = 0
    tracker    = Tracker(max_age=MAX_AGE, min_hits=MIN_HITS,
                         lambda_cost=LAMBDA_COST,
                         gallery_size=GALLERY_SIZE,
                         n_feat_bins=N_FEAT_BINS,
                         gate_chi2=GATE_CHI2,
                         iou_thr=IOU_THR)
    trajectory = []

    sensor_bus = None
    temps = None

    if args.source == "csv":
        temps, _ = load_csv(args.csv)
        n_frames  = len(temps)
        t_min, t_max = float(temps.min()), float(temps.max())
    else:
        from mlx90642_io import open_bus, read_frame  # noqa: WPS433
        sensor_bus = open_bus()
        n_frames = int(1e9)
        t_min, t_max = 20., 60.

    set_temp_range(t_min, t_max)
    display = _Display(args)

    delay = 1.0 / FPS_TARGET
    t_start = time.perf_counter()
    fps_disp = 0.0

    try:
        for fidx in range(n_frames):
            t0 = time.perf_counter()
            if args.source == "csv":
                flat = temps[fidx]
                f2d  = flat.reshape(FRAME_H, FRAME_W)
            else:
                f2d = read_frame(sensor_bus)
                # ковзна адаптація шкали для live-стріму
                t_min = 0.9 * t_min + 0.1 * float(f2d.min())
                t_max = 0.9 * t_max + 0.1 * float(f2d.max())
                set_temp_range(t_min, t_max)

            mask, bbox_mrf = segment_frame(f2d, k_sigma=K_SIGMA, lam=MRF_LAMBDA)
            dets = [bbox_mrf] if bbox_mrf is not None else []

            tracker.predict_all()
            tracker.update(dets, f2d)

            b = tracker.get_best()
            if b:
                r0, r1, c0, c1 = b.get_bbox()
                trajectory.append((int((c0 + c1) / 2 * SCALE_X),
                                   int((r0 + r1) / 2 * SCALE_Y)))
                if len(trajectory) > 60: trajectory.pop(0)

            bgr = frame_to_bgr(f2d)
            render(bgr, f2d, mask, bbox_mrf, tracker, fidx, n_frames, fps_disp, trajectory)
            display.publish(bgr, fidx)

            elapsed  = time.perf_counter() - t0
            wait_ms  = max(1, int((delay - elapsed) * 1000))
            key = display.wait_key(wait_ms)
            if not display.headless and key in (ord('q'), 27):
                break

            if fidx % 10 == 9:
                fps_disp = 10 / (time.perf_counter() - t_start + 1e-9)
                t_start  = time.perf_counter()

            if args.max_frames and fidx + 1 >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        display.close()
        if sensor_bus is not None:
            sensor_bus.close()
        print("Done.")


if __name__ == "__main__":
    run()
