#!/usr/bin/env python3
"""
Прогнати кадри однієї записаної сесії (sessions/session_*) через усі три
треккери з пакета `trackers` і показати їх паралельно в OpenCV.

Приклад:
    python3 rpi_thermal_tracking_z/compare_trackers_on_session.py sessions/session_1
    python3 rpi_thermal_tracking_z/compare_trackers_on_session.py sessions/session_1 --no-cnn
    python3 rpi_thermal_tracking_z/compare_trackers_on_session.py sessions/session_1 --loop --speed 0.5

Сесія = тека з:
    frames.f32       — float32 LE (N × H × W)
    timestamps.f64   — float64 (N)  monotonic seconds
    meta.json        — {"shape": [H, W], "t_min_global_c", "t_max_global_c", "fps_avg", ...}

Сегментація MRF/GC робиться один раз на кадр (однакові параметри для всіх
трекерів), що дає чесне порівняння тільки трекінгу.

Клавіші:
    Q / Esc    — вихід
    SPACE      — пауза/продовжити
    R          — скинути всі треккери
    .          — крок вперед (у режимі паузи)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thermal_viz import colorize  # noqa: E402
from trackers.segmentation import (  # noqa: E402
    segment_frame, DEFAULT_K_SIGMA, DEFAULT_LAMBDA,
)
from trackers.sort import SORTTrack, SORTTracker  # noqa: E402
from trackers.deepsort_temp import (  # noqa: E402
    DeepSORTTempTrack, DeepSORTTempTracker, set_temp_range,
)

PANEL_W, PANEL_H = 480, 360            # розмір однієї панелі
TITLE_BAR_H      = 28                  # висота заголовка панелі
TRAJ_MAX         = 60

TRACK_COLORS = [
    (0, 220, 220), (220, 0, 220), (220, 220, 0),
    (0, 150, 255), (255, 100, 0), (100, 255, 0),
    (0, 255, 150), (150, 0, 255),
]

WINDOW_NAME = "Thermal trackers — SORT vs DeepSORT(temp) vs DeepSORT(CNN)"


# ── завантаження сесії ────────────────────────────────────
def load_session(sdir: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    meta_path = os.path.join(sdir, "meta.json")
    f32_path  = os.path.join(sdir, "frames.f32")
    ts_path   = os.path.join(sdir, "timestamps.f64")
    for p in (meta_path, f32_path, ts_path):
        if not os.path.isfile(p):
            print(f"missing: {p}", file=sys.stderr); sys.exit(1)

    with open(meta_path) as fh:
        meta = json.load(fh)
    shape = tuple(meta["shape"])                       # [H, W]
    frames = np.fromfile(f32_path, dtype="float32").reshape(-1, *shape)
    ts     = np.fromfile(ts_path,  dtype="float64")
    n = min(len(frames), len(ts))
    return frames[:n], ts[:n], meta


# ── рендер однієї панелі ──────────────────────────────────
def _render_panel(frame2d: np.ndarray,
                  mask: np.ndarray | None,
                  bbox_mrf: tuple | None,
                  tracker,
                  title: str,
                  t_min: float, t_max: float,
                  trajectory: list[tuple[int, int]],
                  hud_lines: list[str]) -> np.ndarray:
    H, W = frame2d.shape
    scale_x = PANEL_W / W
    scale_y = PANEL_H / H

    bgr = colorize(frame2d, t_min, t_max, PANEL_W, PANEL_H,
                   interp=cv2.INTER_NEAREST)

    if mask is not None:
        mb = cv2.resize(mask.astype(np.uint8) * 255,
                        (PANEL_W, PANEL_H), interpolation=cv2.INTER_NEAREST)
        ov = bgr.copy()
        ov[mb > 0] = (ov[mb > 0] * 0.6 + np.array([180, 120, 0]) * 0.4).astype(np.uint8)
        bgr[:] = ov

    def tod(r, c):
        return int(c * scale_x), int(r * scale_y)

    if bbox_mrf is not None:
        r0, r1, c0, c1 = bbox_mrf
        cv2.rectangle(bgr, tod(r0, c0), tod(r1, c1), (0, 220, 0), 1)
        cv2.putText(bgr, "MRF", (tod(r0, c0)[0] + 2, max(tod(r0, c0)[1] - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1, cv2.LINE_AA)

    for t in tracker.tracks:
        r0, r1, c0, c1 = t.get_bbox()
        x0, y0 = tod(r0, c0); x1, y1 = tod(r1, c1)
        x0, x1 = max(0, x0), min(PANEL_W - 1, x1)
        y0, y1 = max(0, y0), min(PANEL_H - 1, y1)
        clr   = TRACK_COLORS[(t.track_id - 1) % len(TRACK_COLORS)]
        thick = 2 if t.confirmed else 1
        cv2.rectangle(bgr, (x0, y0), (x1, y1), clr, thick)
        gal = len(getattr(t, "gallery", []) or [])
        label = f"ID{t.track_id} h{t.hits}" + (f" g{gal}" if gal else "")
        cv2.putText(bgr, label, (x0, max(y0 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, clr, 1, cv2.LINE_AA)
        vx, vy = t.get_velocity()
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        dx, dy = int(vx * scale_x * 3), int(vy * scale_y * 3)
        if abs(dx) + abs(dy) > 2:
            cv2.arrowedLine(bgr, (cx, cy), (cx + dx, cy + dy),
                            clr, 2, tipLength=0.3)

    for j in range(1, len(trajectory)):
        a = j / len(trajectory)
        cv2.line(bgr, trajectory[j - 1], trajectory[j],
                 (int(255 * a), int(200 * a), 0), 1)

    for i, line in enumerate(hud_lines):
        yp = 18 + i * 18
        cv2.putText(bgr, line, (6, yp), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(bgr, line, (6, yp), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # титульний бар
    title_bg = np.full((TITLE_BAR_H, PANEL_W, 3), 32, dtype=np.uint8)
    cv2.putText(title_bg, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([title_bg, bgr])


def _update_trajectory(traj: list[tuple[int, int]], tracker, w: int, h: int) -> None:
    best = tracker.get_best()
    if best is None:
        return
    r0, r1, c0, c1 = best.get_bbox()
    traj.append((int((c0 + c1) / 2 * PANEL_W / w),
                 int((r0 + r1) / 2 * PANEL_H / h)))
    if len(traj) > TRAJ_MAX:
        traj.pop(0)


# ── main ──────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Прогон записаної сесії через SORT / DeepSORT(temp) / DeepSORT(CNN)")
    p.add_argument("session_dir",
                   help="тека з frames.f32 / timestamps.f64 / meta.json")
    p.add_argument("--model",
                   default=os.path.join(
                       os.path.dirname(os.path.abspath(__file__)),
                       "micro_reid_ir.pt"),
                   help="шлях до чекпойнта MicroReIDIR (для CNN-трекера)")
    p.add_argument("--no-cnn", action="store_true",
                   help="не запускати CNN-трекер (якщо torch недоступний)")
    p.add_argument("--scale-mode",
                   choices=["fixed-meta", "auto", "robust"],
                   default="fixed-meta",
                   help="як вибирати T_min/T_max для палітри: "
                        "fixed-meta (з meta.json), auto (per-frame min/max), "
                        "robust (2%/98% percentile по всій сесії)")
    p.add_argument("--k-sigma", type=float, default=DEFAULT_K_SIGMA,
                   help="поріг сегментації mean + K*std")
    p.add_argument("--lam", type=float, default=DEFAULT_LAMBDA,
                   help="вага регуляризації MRF")
    p.add_argument("--speed", type=float, default=1.0,
                   help="множник швидкості відтворення (1.0 = реальний час)")
    p.add_argument("--loop", action="store_true",
                   help="зациклити відтворення")
    p.add_argument("--save", default=None,
                   help="зберегти склейку трьох панелей як MP4 за цим шляхом")
    return p.parse_args()


def _t_range(scale_mode: str, frames: np.ndarray,
             meta: dict[str, Any]) -> tuple[float, float] | None:
    """Повертає (t_min, t_max) або None якщо шкалу треба брати з кадру."""
    if scale_mode == "fixed-meta":
        return (float(meta.get("t_min_global_c") or frames.min()),
                float(meta.get("t_max_global_c") or frames.max()))
    if scale_mode == "robust":
        return (float(np.percentile(frames, 2)),
                float(np.percentile(frames, 98)))
    return None  # auto


def main() -> None:
    args = _parse_args()
    frames, ts, meta = load_session(args.session_dir)
    H, W = frames.shape[1:]
    print(f"[compare] session={args.session_dir}  frames={len(frames)}  shape=({H},{W})")

    if len(frames) < 2:
        print("not enough frames"); return

    duration = float(ts[-1] - ts[0])
    fps_native = len(frames) / duration if duration > 0 else float(meta.get("fps_avg", 8))
    target_fps = max(0.1, fps_native * args.speed)
    delay      = 1.0 / target_fps
    print(f"[compare] native_fps={fps_native:.2f}  target_fps={target_fps:.2f}  delay={delay:.3f}s")

    fixed_range = _t_range(args.scale_mode, frames, meta)
    if fixed_range is None:
        # для нормалізації CNN-патчу однаково потрібен якийсь діапазон;
        # беремо глобальний як стартову точку
        cnn_range = (float(frames.min()), float(frames.max()))
    else:
        cnn_range = fixed_range
    print(f"[compare] scale_mode={args.scale_mode}  cnn_range={cnn_range}")

    # ── треккери ──────────────────────────────────────────
    SORTTrack._id_counter = 0
    DeepSORTTempTrack._id_counter = 0
    set_temp_range(*cnn_range)
    sort_tr = SORTTracker()
    temp_tr = DeepSORTTempTracker()
    panels: list[tuple[str, object, list[tuple[int, int]]]] = [
        ("SORT (Kalman + IoU)",            sort_tr, []),
        ("DeepSORT (temp-histogram ReID)", temp_tr, []),
    ]

    cnn_tr = None
    if not args.no_cnn:
        try:
            from trackers.deepsort_cnn import (  # noqa: WPS433
                DeepSORTCNNTrack, DeepSORTCNNTracker, load_model,
            )
            if not os.path.isfile(args.model):
                print(f"[compare] WARN: model not found: {args.model} — CNN-трекер вимкнено")
            else:
                DeepSORTCNNTrack._id_counter = 0
                model, edim = load_model(args.model)
                cnn_tr = DeepSORTCNNTracker(model, edim, *cnn_range)
                panels.append(("DeepSORT (MicroReIDIR CNN)", cnn_tr, []))
                print(f"[compare] CNN model loaded: {args.model} embed_dim={edim}")
        except ImportError as e:
            print(f"[compare] CNN-трекер пропущено (torch недоступний): {e}")

    # ── вікно ─────────────────────────────────────────────
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    full_w = PANEL_W * len(panels)
    full_h = PANEL_H + TITLE_BAR_H
    cv2.resizeWindow(WINDOW_NAME, full_w, full_h)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, target_fps, (full_w, full_h))
        if not writer.isOpened():
            print(f"[compare] WARN: cannot open {args.save}")
            writer = None
        else:
            print(f"[compare] writing {args.save}  size=({full_w},{full_h})  fps={target_fps:.2f}")

    paused = False
    fidx   = 0
    print("[compare] Q/Esc — вихід, SPACE — пауза, . — крок, R — reset")

    try:
        while True:
            t0 = time.perf_counter()
            frame = frames[fidx]
            ts_now = float(ts[fidx] - ts[0])

            if fixed_range is None:
                t_min, t_max = float(frame.min()), float(frame.max())
            else:
                t_min, t_max = fixed_range

            mask, bbox_mrf = segment_frame(frame, k_sigma=args.k_sigma, lam=args.lam)
            detections = [bbox_mrf] if bbox_mrf is not None else []

            # SORT (frame2d не потрібен)
            sort_tr.predict_all(); sort_tr.update(detections)
            # DeepSORT temp
            temp_tr.predict_all(); temp_tr.update(detections, frame)
            # DeepSORT CNN
            if cnn_tr is not None:
                cnn_tr.predict_all(); cnn_tr.update(detections, frame)

            for _title, tr, traj in panels:
                _update_trajectory(traj, tr, W, H)

            hud_common = [
                f"frame {fidx:04d}/{len(frames) - 1}  t={ts_now:5.2f}s",
                f"Tmin={frame.min():.1f}C  Tmax={frame.max():.1f}C",
            ]

            panel_images = []
            for title, tr, traj in panels:
                best   = tr.get_best()
                active = tr.get_active() if hasattr(tr, "get_active") \
                    else tr.get_active_tracks()
                hud = list(hud_common) + [
                    f"tracks {len(tr.tracks)}  conf {len(active)}",
                    f"best ID {best.track_id if best else -1}",
                ]
                panel_images.append(
                    _render_panel(frame, mask, bbox_mrf, tr, title,
                                  t_min, t_max, traj, hud))

            combined = np.hstack(panel_images)
            cv2.imshow(WINDOW_NAME, combined)
            if writer is not None:
                writer.write(combined)

            # таймінг
            elapsed = time.perf_counter() - t0
            wait_ms = max(1, int((delay - elapsed) * 1000)) if not paused else 0
            key = cv2.waitKey(wait_ms) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                print(f"[compare] {'pause' if paused else 'play'}")
            if key == ord("r"):
                sort_tr.reset(); temp_tr.reset()
                if cnn_tr is not None:
                    cnn_tr.reset()
                for _t, _tr, traj in panels:
                    traj.clear()
                print("[compare] reset")
            step_one = (key == ord("."))

            if paused and not step_one:
                continue

            fidx += 1
            if fidx >= len(frames):
                if args.loop:
                    fidx = 0
                    sort_tr.reset(); temp_tr.reset()
                    if cnn_tr is not None:
                        cnn_tr.reset()
                    for _t, _tr, traj in panels:
                        traj.clear()
                    print("[compare] loop restart")
                else:
                    break
    except KeyboardInterrupt:
        print("\n[compare] interrupted")
    finally:
        if writer is not None:
            writer.release()
            print(f"[compare] wrote {args.save}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
