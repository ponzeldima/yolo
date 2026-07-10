#!/usr/bin/env python3
"""
MLX90642 recorder з керуванням через браузер.

Поведінка:
  * Скрипт постійно читає сенсор і ВЕСЬ ЧАС стрімить у браузер.
  * Записування вмикається / вимикається кнопками на сторінці:
      START REC  → починаємо нову сесію у session_YYYYMMDD_HHMMSS/
      STOP  REC  → закриваємо файли поточної сесії, стрім продовжується.
      Quit       → завершує скрипт.
  * Можна знову натиснути START — буде нова сесія з новим іменем.
  * У терміналі Enter = Quit, Ctrl-C теж працює.

Формат сесії (тека session_YYYYMMDD_HHMMSS/):
  - frames.f32      сирий потік float32 LE: N × 24 × 32
  - timestamps.f64  monotonic секунди, по float64 на кадр
  - video.mp4       візуалізація (inferno LUT) для перегляду
  - frames.csv      опційно (--csv), сумісний з трекерами керівника
  - meta.json       shape, dtype, n_frames, fps_avg, ...

Завантаження назад:
    import numpy as np, json
    m = json.load(open('meta.json'))
    f = np.fromfile('frames.f32', 'float32').reshape(-1, *m['shape'])
    t = np.fromfile('timestamps.f64', 'float64')
"""
from __future__ import annotations

import argparse
import csv as csvmod
import datetime as dt
import json
import os
import select
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mjpeg_server import start_mjpeg_server, lan_ip_hint  # noqa: E402
from mlx90642_io import open_bus, read_frame, FRAME_H, FRAME_W  # noqa: E402
from thermal_viz import colorize  # noqa: E402


# ── Enter у терміналі → quit ───────────────────────────────
_QUIT_FROM_STDIN = threading.Event()


def _stdin_watcher() -> None:
    try:
        while not _QUIT_FROM_STDIN.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.5)
            if r:
                sys.stdin.readline()
                _QUIT_FROM_STDIN.set()
                return
    except (OSError, ValueError):
        return


# ── Контекст однієї сесії запису ───────────────────────────
class _RecSession:
    """Тримає всі open-файли + лічильники для однієї сесії запису."""

    def __init__(self, out_dir: str, disp_w: int, disp_h: int,
                 write_video: bool, write_csv: bool,
                 video_fps_initial: float) -> None:
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.disp_w = disp_w
        self.disp_h = disp_h
        self.f32 = open(os.path.join(out_dir, "frames.f32"), "wb",
                        buffering=0)
        self.ts  = open(os.path.join(out_dir, "timestamps.f64"), "wb",
                        buffering=0)
        self.csv_f = None
        self.csv_w = None
        if write_csv:
            self.csv_f = open(os.path.join(out_dir, "frames.csv"), "w",
                              newline="")
            self.csv_w = csvmod.writer(self.csv_f, delimiter=";")
        self.video: cv2.VideoWriter | None = None
        if write_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video = cv2.VideoWriter(
                os.path.join(out_dir, "video.mp4"),
                fourcc, video_fps_initial, (disp_w, disp_h))
            if not self.video.isOpened():
                print(f"[rec] WARN: cannot open video.mp4 in {out_dir}")
                self.video = None
        self.video_fps_initial = video_fps_initial
        self.n_frames = 0
        self.t_min_global = +1e9
        self.t_max_global = -1e9
        self.t_start = time.perf_counter()

    def write(self, frame: np.ndarray, bgr: np.ndarray,
              ts_now: float) -> None:
        self.f32.write(frame.astype(np.float32).tobytes(order="C"))
        self.ts.write(np.float64(ts_now).tobytes())
        if self.csv_w is not None:
            iso = dt.datetime.now().isoformat(timespec="milliseconds")
            self.csv_w.writerow([iso] + [f"{v:.2f}" for v in frame.ravel()])
        if self.video is not None:
            self.video.write(bgr)
        fmin, fmax = float(frame.min()), float(frame.max())
        if fmin < self.t_min_global: self.t_min_global = fmin
        if fmax > self.t_max_global: self.t_max_global = fmax
        self.n_frames += 1

    def close(self) -> dict:
        elapsed = time.perf_counter() - self.t_start
        fps_avg = self.n_frames / elapsed if elapsed > 0 else 0.0
        self.f32.close()
        self.ts.close()
        if self.csv_f is not None:
            self.csv_f.close()
        if self.video is not None:
            self.video.release()
        meta = {
            "shape": [FRAME_H, FRAME_W],
            "dtype": "float32",
            "endian": "little",
            "n_frames": self.n_frames,
            "duration_s": round(elapsed, 3),
            "fps_avg": round(fps_avg, 3),
            "video_fps_used": self.video_fps_initial,
            "t_min_global_c": round(self.t_min_global, 3)
                              if self.n_frames else None,
            "t_max_global_c": round(self.t_max_global, 3)
                              if self.n_frames else None,
            "files": {
                "frames": "frames.f32",
                "timestamps": "timestamps.f64",
                "csv": "frames.csv" if self.csv_w is not None else None,
                "video": "video.mp4" if self.video is not None else None,
            },
        }
        with open(os.path.join(self.out_dir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        return meta


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "MLX90642 recorder (stream завжди, запис вмикається з браузера)")
    p.add_argument("--out-root", default=".",
                   help="батьківська тека для session_YYYYMMDD_HHMMSS/")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--disp-w", type=int, default=640)
    p.add_argument("--disp-h", type=int, default=480)
    p.add_argument("--no-video", action="store_true",
                   help="не писати video.mp4 у сесії")
    p.add_argument("--csv", action="store_true",
                   help="додатково писати frames.csv у сесії")
    p.add_argument("--video-fps", type=float, default=10.0,
                   help="FPS для video.mp4 (стартова догадка)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── MJPEG-сервер (стрім завжди) ───────────────────────
    bus, httpd = start_mjpeg_server(args.host, args.port)
    ip = lan_ip_hint() if args.host in ("0.0.0.0", "") else args.host
    bus.set_status("IDLE")
    print(f"[rec] open in browser: http://{ip}:{args.port}/")
    print(f"[rec] кнопки: START REC / STOP REC / Quit")
    print(f"[rec] (Enter у цьому терміналі = Quit; Ctrl-C теж працює)")

    threading.Thread(target=_stdin_watcher, daemon=True).start()

    # ── Сенсор ────────────────────────────────────────────
    sensor_bus = open_bus()
    print(f"[rec] sensor opened")

    # ── Стан ──────────────────────────────────────────────
    session: _RecSession | None = None
    t_min_disp = 20.0
    t_max_disp = 60.0
    n_total = 0
    t_log = time.perf_counter()
    n_log = 0

    try:
        while True:
            # ── обробка керуючих сигналів з браузера ─────
            if bus.consume("start"):
                if session is None:
                    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_dir = os.path.join(args.out_root, f"session_{stamp}")
                    session = _RecSession(
                        out_dir, args.disp_w, args.disp_h,
                        write_video=not args.no_video,
                        write_csv=args.csv,
                        video_fps_initial=args.video_fps)
                    print(f"[rec] START → {out_dir}/")
                else:
                    print("[rec] START ignored (вже записує)")
            if bus.consume("stop"):
                if session is not None:
                    meta = session.close()
                    print(f"[rec] STOP → {session.out_dir}/ "
                          f"frames={meta['n_frames']} "
                          f"fps_avg={meta['fps_avg']}")
                    session = None
                    bus.set_status("IDLE")
                # коли session is None, STOP просто ігнорується,
                # стрім продовжується (Quit окремий)
            if bus.consume("quit") or _QUIT_FROM_STDIN.is_set():
                print("[rec] Quit")
                break

            # ── читання сенсора (tight loop, без sleep) ──
            try:
                frame = read_frame(sensor_bus)
            except OSError as exc:
                print(f"[rec] I2C read error: {exc!r}")
                continue
            ts_now = time.perf_counter()
            n_total += 1

            # ковзна шкала температур для красивого стріму
            fmin, fmax = float(frame.min()), float(frame.max())
            t_min_disp = 0.9 * t_min_disp + 0.1 * fmin
            t_max_disp = 0.9 * t_max_disp + 0.1 * fmax

            bgr = colorize(frame, t_min_disp, t_max_disp,
                           args.disp_w, args.disp_h)
            cv2.putText(
                bgr,
                f"T={fmin:.1f}..{fmax:.1f}C  "
                f"frames={n_total}  "
                f"{'REC' if session else 'idle'}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

            # ── запис (якщо активна сесія) ──────────────
            if session is not None:
                session.write(frame, bgr, ts_now)
                bus.set_status(
                    f"REC ●  {session.n_frames:5d}  "
                    f"T=[{session.t_min_global:.1f}.."
                    f"{session.t_max_global:.1f}]C")
            else:
                bus.set_status(
                    f"IDLE  total={n_total}  "
                    f"T=[{fmin:.1f}..{fmax:.1f}]C")

            # ── MJPEG публікація ─────────────────────────
            ok, jpg = cv2.imencode(
                ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if ok:
                bus.publish(jpg.tobytes())

            # ── періодичний лог fps ──────────────────────
            if ts_now - t_log >= 5.0:
                fps_now = (n_total - n_log) / (ts_now - t_log)
                state = "REC" if session else "idle"
                print(f"[rec] {state}  total={n_total}  fps={fps_now:.2f}")
                t_log = ts_now
                n_log = n_total

    except KeyboardInterrupt:
        print("\n[rec] Ctrl-C")
    finally:
        if session is not None:
            meta = session.close()
            print(f"[rec] final STOP → {session.out_dir}/ "
                  f"frames={meta['n_frames']}")
        bus.stop()
        httpd.shutdown()
        sensor_bus.close()
        print(f"[rec] done; total frames seen = {n_total}")


if __name__ == "__main__":
    main()
