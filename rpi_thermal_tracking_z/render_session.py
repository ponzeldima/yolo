#!/usr/bin/env python3
"""
Рендерить теплову сесію (frames.f32 + timestamps.f64 + meta.json) у MP4-відео
у різних палітрах і режимах нормалізації.

Приклад:
    python3 render_session.py sessions/session_1
    python3 render_session.py sessions/session_1 --hud
    python3 render_session.py sessions/session_1 \\
        --palettes inferno,turbo,white_hot --modes auto,robust

Виходи: <session_dir>/<palette>_<mode>.mp4

FPS відео = n_frames / (timestamps[-1] - timestamps[0]) — точно відповідає
реальній швидкості зйомки.

Палітри:
    inferno, viridis, turbo, hot, ironbow, white_hot

Режими нормалізації:
    auto    — min/max поточного кадру (стрибає)
    fixed   — константи [--t-min, --t-max] (за замовч. глобальні з meta)
    robust  — percentile 2%/98% по всьому датасету (одна шкала на відео)
    agc     — histogram equalization кожного кадру (як у військових тепловіз.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable

import cv2
import numpy as np


# ── Палітри ───────────────────────────────────────────────
_INFERNO_KEY = np.array([
    [0.001462, 0.000466, 0.013866],
    [0.087411, 0.044556, 0.224813],
    [0.258234, 0.038571, 0.406485],
    [0.416331, 0.090203, 0.432943],
    [0.578375, 0.148039, 0.404409],
    [0.735683, 0.215906, 0.330245],
    [0.865006, 0.316822, 0.226055],
    [0.955834, 0.468780, 0.099874],
    [0.993248, 0.652325, 0.038965],
    [0.964394, 0.843848, 0.273391],
    [0.988362, 0.998364, 0.644924],
], dtype=np.float32)

# FLIR-style ironbow keypoints у RGB
_IRONBOW_KEY = np.array([
    [0.00, 0.00, 0.00],
    [0.05, 0.00, 0.20],
    [0.20, 0.00, 0.45],
    [0.45, 0.05, 0.55],
    [0.70, 0.10, 0.45],
    [0.90, 0.25, 0.30],
    [1.00, 0.45, 0.10],
    [1.00, 0.70, 0.00],
    [1.00, 0.90, 0.20],
    [1.00, 1.00, 0.85],
], dtype=np.float32)


def _build_lut_bgr(key_rgb: np.ndarray) -> np.ndarray:
    t = np.linspace(0, 1, 256)
    tk = np.linspace(0, 1, len(key_rgb))
    rgb = np.stack([np.interp(t, tk, key_rgb[:, i]) for i in range(3)], axis=1)
    bgr = rgb[:, ::-1]
    return (bgr * 255).clip(0, 255).astype(np.uint8)


_LUT_INFERNO = _build_lut_bgr(_INFERNO_KEY)
_LUT_IRONBOW = _build_lut_bgr(_IRONBOW_KEY)


def apply_palette(idx: np.ndarray, palette: str) -> np.ndarray:
    """idx (H,W) uint8 → (H,W,3) BGR uint8."""
    if palette == "inferno":
        return _LUT_INFERNO[idx]
    if palette == "ironbow":
        return _LUT_IRONBOW[idx]
    if palette == "white_hot":
        return cv2.cvtColor(idx, cv2.COLOR_GRAY2BGR)
    cmap = {
        "viridis": cv2.COLORMAP_VIRIDIS,
        "turbo":   cv2.COLORMAP_TURBO,
        "hot":     cv2.COLORMAP_HOT,
    }[palette]
    return cv2.applyColorMap(idx, cmap)


PALETTES = ["inferno", "viridis", "turbo", "hot", "ironbow", "white_hot"]
MODES    = ["auto", "fixed", "robust", "agc"]


# ── Нормалізація кадру → uint8 idx ─────────────────────────
def _to_idx_linear(frame: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    span = max(t_max - t_min, 1e-3)
    return ((frame - t_min) / span * 255).clip(0, 255).astype(np.uint8)


def make_idx_func(mode: str, frames: np.ndarray, args: argparse.Namespace,
                  meta: dict) -> tuple[Callable[[np.ndarray], np.ndarray],
                                       tuple[float, float] | None]:
    """Повертає функцію (frame → idx uint8) і (t_min, t_max) для legend
    (None якщо шкала змінюється покадрово)."""
    if mode == "auto":
        def fn(frame: np.ndarray) -> np.ndarray:
            return _to_idx_linear(frame, float(frame.min()),
                                  float(frame.max()))
        return fn, None

    if mode == "fixed":
        t_min = (args.t_min if args.t_min is not None
                 else meta.get("t_min_global_c", 20.0))
        t_max = (args.t_max if args.t_max is not None
                 else meta.get("t_max_global_c", 60.0))
        rng = (float(t_min), float(t_max))

        def fn(frame: np.ndarray) -> np.ndarray:
            return _to_idx_linear(frame, rng[0], rng[1])
        return fn, rng

    if mode == "robust":
        lo = float(np.percentile(frames, 2))
        hi = float(np.percentile(frames, 98))
        rng = (lo, hi)

        def fn(frame: np.ndarray) -> np.ndarray:
            return _to_idx_linear(frame, rng[0], rng[1])
        return fn, rng

    if mode == "agc":
        # spread кадру в [0,255] лінійно по його min/max,
        # потім histogram equalization.
        def fn(frame: np.ndarray) -> np.ndarray:
            idx = _to_idx_linear(frame, float(frame.min()),
                                 float(frame.max()))
            return cv2.equalizeHist(idx)
        return fn, None

    raise ValueError(f"unknown mode: {mode}")


# ── Рендер одного відео ────────────────────────────────────
def render_one(frames: np.ndarray, palette: str, mode: str, fps: float,
               disp_w: int, disp_h: int, out_path: str,
               args: argparse.Namespace, meta: dict) -> None:
    idx_fn, rng = make_idx_func(mode, frames, args, meta)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (disp_w, disp_h))
    if not writer.isOpened():
        print(f"  ! cannot open {out_path}")
        return

    legend = f"{palette} / {mode}"
    if rng is not None:
        legend += f" [{rng[0]:.1f}..{rng[1]:.1f} C]"

    for frame in frames:
        idx_small = idx_fn(frame)
        bgr_small = apply_palette(idx_small, palette)
        bgr = cv2.resize(bgr_small, (disp_w, disp_h),
                         interpolation=cv2.INTER_NEAREST)
        if args.hud:
            cv2.putText(bgr, legend, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(bgr, legend, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(bgr)
    writer.release()
    size_kb = os.path.getsize(out_path) / 1024.0
    print(f"  ok  {os.path.basename(out_path):28s} "
          f"{size_kb:7.1f} KB  {legend}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Render thermal session in many palettes/modes")
    p.add_argument("session_dir",
                   help="тека з frames.f32 / timestamps.f64 / meta.json")
    p.add_argument("--palettes", default=",".join(PALETTES),
                   help=f"comma-separated; доступні: {','.join(PALETTES)}")
    p.add_argument("--modes", default=",".join(MODES),
                   help=f"comma-separated; доступні: {','.join(MODES)}")
    p.add_argument("--t-min", type=float, default=None,
                   help="нижня межа для fixed-режиму (за замовч. з meta)")
    p.add_argument("--t-max", type=float, default=None,
                   help="верхня межа для fixed-режиму (за замовч. з meta)")
    p.add_argument("--disp-w", type=int, default=640)
    p.add_argument("--disp-h", type=int, default=480)
    p.add_argument("--hud", action="store_true",
                   help="накладати підпис палітра/режим/діапазон")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    sdir = args.session_dir
    meta_path = os.path.join(sdir, "meta.json")
    f32_path  = os.path.join(sdir, "frames.f32")
    ts_path   = os.path.join(sdir, "timestamps.f64")
    for path in (meta_path, f32_path, ts_path):
        if not os.path.isfile(path):
            print(f"missing: {path}", file=sys.stderr)
            sys.exit(1)

    with open(meta_path) as fh:
        meta = json.load(fh)
    shape = tuple(meta["shape"])               # [H, W]
    frames = np.fromfile(f32_path, dtype="float32").reshape(-1, *shape)
    ts     = np.fromfile(ts_path,  dtype="float64")
    if len(frames) != len(ts):
        print(f"WARN: n_frames mismatch  frames={len(frames)} ts={len(ts)}")
        n = min(len(frames), len(ts))
        frames = frames[:n]; ts = ts[:n]

    if len(frames) < 2:
        print("not enough frames"); sys.exit(1)

    duration = float(ts[-1] - ts[0])
    fps = len(frames) / duration if duration > 0 else float(meta.get("fps_avg", 8))

    palettes = [p.strip() for p in args.palettes.split(",") if p.strip()]
    modes    = [m.strip() for m in args.modes.split(",")    if m.strip()]
    bad = [p for p in palettes if p not in PALETTES] + \
          [m for m in modes    if m not in MODES]
    if bad:
        print(f"unknown: {bad}\n  palettes: {PALETTES}\n  modes: {MODES}")
        sys.exit(1)

    print(f"[render] {sdir}")
    print(f"  frames     : {len(frames)}")
    print(f"  shape      : {shape}")
    print(f"  duration_s : {duration:.3f}")
    print(f"  fps        : {fps:.3f}")
    print(f"  palettes   : {palettes}")
    print(f"  modes      : {modes}")
    print(f"  hud        : {args.hud}")
    print(f"  total mp4s : {len(palettes) * len(modes)}")
    print()

    for palette in palettes:
        for mode in modes:
            out = os.path.join(sdir, f"{palette}_{mode}.mp4")
            render_one(frames, palette, mode, fps,
                       args.disp_w, args.disp_h, out, args, meta)

    print("\n[render] done")


if __name__ == "__main__":
    main()
