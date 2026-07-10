#!/usr/bin/env python3
"""
Підготувати сесію (sessions/session_*) для розмітки в CVAT.

Що робить:
  1) Читає frames.f32 / timestamps.f64 / meta.json.
  2) Пише `labeling.mp4` — upscale кадрів nearest-neighbor у роздільність, зручну
     для миші (за замовч. 320×240 = ×10 від 32×24). Жодного blur, межі пікселів
     зберігаються — щоб у CVAT можна було рамкою точно "пристрелятися" до пікселя.
  3) Пише `cvat_scale.json` — масштаб + усі параметри, потрібні
     `cvat_to_gt.py`, щоб поділити координати назад у 24×32.

Приклад:
    python3 rpi_thermal_tracking_z/cvat_prepare.py sessions/session_5
    python3 rpi_thermal_tracking_z/cvat_prepare.py sessions/session_5 \\
        --palette inferno --scale-mode robust --upscale 16

Що далі робити в CVAT (короткий чек-лист у кінці stdout):
  - Створити Task → Video → залити sessions/session_5/cvat/labeling.mp4.
  - Frame Step = 1, FPS = той, що в meta.
  - Додати label "cup" (rectangle), розмітити (можна з лінійною інтерполяцією).
  - Export → "MOT 1.1" → ZIP → розпакувати, зберегти `gt.txt`.
  - Прогнати `cvat_to_gt.py` (див. його хелп).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_session import (  # noqa: E402
    PALETTES, MODES, apply_palette, make_idx_func,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Підготувати сесію MLX90642 до розмітки в CVAT")
    p.add_argument("session_dir",
                   help="тека з frames.f32 / timestamps.f64 / meta.json")
    p.add_argument("--out-dir", default=None,
                   help="куди класти labeling.mp4 (за замовч. <session_dir>/cvat/)")
    p.add_argument("--palette", default="inferno", choices=PALETTES,
                   help="палітра для відео розмітки")
    p.add_argument("--scale-mode", default="robust", choices=MODES,
                   help="режим нормалізації (robust = стабільна шкала по всій сесії; "
                        "рекомендовано для розмітки)")
    p.add_argument("--upscale", type=int, default=10,
                   help="у скільки разів збільшити кадр (24×32 → 240×320 при ×10)")
    p.add_argument("--t-min", type=float, default=None,
                   help="нижня межа для scale-mode=fixed (за замовч. з meta)")
    p.add_argument("--t-max", type=float, default=None,
                   help="верхня межа для scale-mode=fixed (за замовч. з meta)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    sdir = args.session_dir
    meta_path = os.path.join(sdir, "meta.json")
    f32_path  = os.path.join(sdir, "frames.f32")
    ts_path   = os.path.join(sdir, "timestamps.f64")
    for path in (meta_path, f32_path, ts_path):
        if not os.path.isfile(path):
            print(f"missing: {path}", file=sys.stderr); sys.exit(1)

    with open(meta_path) as fh:
        meta = json.load(fh)
    shape = tuple(meta["shape"])                 # [H, W] = [24, 32]
    H_orig, W_orig = int(shape[0]), int(shape[1])
    frames = np.fromfile(f32_path, dtype="float32").reshape(-1, *shape)
    ts     = np.fromfile(ts_path,  dtype="float64")
    n = min(len(frames), len(ts))
    frames, ts = frames[:n], ts[:n]
    if len(frames) < 1:
        print("not enough frames", file=sys.stderr); sys.exit(1)

    duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    fps_native = (len(frames) / duration) if duration > 0 \
        else float(meta.get("fps_avg", 8))

    up = max(1, int(args.upscale))
    label_w = W_orig * up
    label_h = H_orig * up

    out_dir = args.out_dir or os.path.join(sdir, "cvat")
    os.makedirs(out_dir, exist_ok=True)
    out_mp4   = os.path.join(out_dir, "labeling.mp4")
    out_scale = os.path.join(out_dir, "cvat_scale.json")

    # adapter for render_session.make_idx_func (потрібен argparse-namespace)
    idx_args = argparse.Namespace(t_min=args.t_min, t_max=args.t_max, hud=False)
    idx_fn, rng = make_idx_func(args.scale_mode, frames, idx_args, meta)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, fps_native, (label_w, label_h))
    if not writer.isOpened():
        print(f"cannot open {out_mp4}", file=sys.stderr); sys.exit(1)

    for frame in frames:
        idx_small = idx_fn(frame)                          # (H, W) uint8
        bgr_small = apply_palette(idx_small, args.palette) # (H, W, 3) BGR
        bgr = cv2.resize(bgr_small, (label_w, label_h),
                         interpolation=cv2.INTER_NEAREST)
        writer.write(bgr)
    writer.release()

    scale_info = {
        "session_dir":     os.path.abspath(sdir),
        "video":           os.path.basename(out_mp4),
        "n_frames":        len(frames),
        "fps":             round(fps_native, 3),
        "orig_w":          W_orig,        # 32
        "orig_h":          H_orig,        # 24
        "label_w":         label_w,       # 320 при upscale=10
        "label_h":         label_h,       # 240 при upscale=10
        "scale_x":         label_w / W_orig,
        "scale_y":         label_h / H_orig,
        "palette":         args.palette,
        "scale_mode":      args.scale_mode,
        "t_range_used":    list(rng) if rng is not None else None,
        "frame_index_base": 1,            # CVAT MOT 1.1 використовує 1-based frame index
    }
    with open(out_scale, "w") as fh:
        json.dump(scale_info, fh, indent=2)

    size_kb = os.path.getsize(out_mp4) / 1024.0
    print(f"[cvat] wrote {out_mp4}  ({size_kb:.1f} KB)")
    print(f"[cvat] wrote {out_scale}")
    print(f"[cvat] resolution для CVAT: {label_w}x{label_h}  "
          f"(×{up} від оригінальних {W_orig}x{H_orig})")
    print(f"[cvat] FPS: {fps_native:.2f}  frames: {len(frames)}")
    print()
    print("──── Що робити в CVAT ─────────────────────────────────")
    print(f" 1. Create Task → залити {out_mp4}")
    print(" 2. Add label, наприклад 'cup' (type: rectangle)")
    print(" 3. Розмітити перший кадр, де об'єкт з'явився,")
    print("    та останній; CVAT інтерполює проміжні keyframe-и")
    print(" 4. Menu → Export task dataset → MOT 1.1 → завантажити ZIP")
    print(" 5. Розпакувати, знайти всередині `gt/gt.txt`,")
    print("    далі прогнати:")
    print(f"      python3 rpi_thermal_tracking_z/cvat_to_gt.py "
          f"<cvat_export>/gt/gt.txt {sdir}")
    print("    скрипт сам поділить координати назад у 24×32 і збереже")
    print(f"    {os.path.join(sdir, 'gt.txt')}")
    print("───────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
