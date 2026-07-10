#!/usr/bin/env python3
"""
Конвертувати експорт CVAT (MOT 1.1) у `gt.txt` із координатами в системі
оригінального сенсора 24×32.

Формат CVAT MOT 1.1 (як і MOTChallenge):
    frame, id, x, y, w, h, conf, class, visibility
де (x, y) — top-left bbox у пікселях відеo CVAT (тобто upscaled).

Цей скрипт:
  * читає cvat_scale.json (створений `cvat_prepare.py`), щоб знати scale_x/scale_y;
  * ділить (x, y, w, h) на scale_x/scale_y → координати в системі сенсора;
  * клампить bbox у межі [0..orig_w-1] × [0..orig_h-1];
  * дозволяє опційно округлити до цілих (для GT це майже завжди потрібно);
  * пише результат у <session_dir>/gt.txt у тому ж MOT-форматі,
    готовий для motmetrics / TrackEval.

Приклад:
    python3 rpi_thermal_tracking_z/cvat_to_gt.py \\
        ~/Downloads/session_5_cvat/gt/gt.txt sessions/session_5

    # перевірити (без запису):
    python3 rpi_thermal_tracking_z/cvat_to_gt.py ... --dry-run --preview 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Конвертер CVAT MOT 1.1 → GT у координатах сенсора 24×32")
    p.add_argument("cvat_gt",
                   help="шлях до gt.txt з ZIP-експорту CVAT MOT 1.1 "
                        "(зазвичай <task>/gt/gt.txt)")
    p.add_argument("session_dir",
                   help="тека сесії (там має лежати cvat/cvat_scale.json)")
    p.add_argument("--scale-json", default=None,
                   help="явний шлях до cvat_scale.json "
                        "(за замовч. <session_dir>/cvat/cvat_scale.json)")
    p.add_argument("--out", default=None,
                   help="куди записати (за замовч. <session_dir>/gt.txt)")
    p.add_argument("--no-round", action="store_true",
                   help="не округлювати координати до int (за замовч. округляємо)")
    p.add_argument("--no-clamp", action="store_true",
                   help="не клампити bbox у межі сенсора")
    p.add_argument("--dry-run", action="store_true",
                   help="нічого не писати, лише надрукувати перші N рядків")
    p.add_argument("--preview", type=int, default=10,
                   help="скільки рядків показати для --dry-run / у summary")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    scale_path = args.scale_json or os.path.join(
        args.session_dir, "cvat", "cvat_scale.json")
    if not os.path.isfile(scale_path):
        print(f"missing scale file: {scale_path}\n"
              "  Запусти спочатку cvat_prepare.py, або вкажи --scale-json.",
              file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.cvat_gt):
        print(f"missing cvat gt: {args.cvat_gt}", file=sys.stderr); sys.exit(1)

    with open(scale_path) as fh:
        scale = json.load(fh)
    scale_x = float(scale["scale_x"])
    scale_y = float(scale["scale_y"])
    W = int(scale["orig_w"]); H = int(scale["orig_h"])
    print(f"[cvt] scale_x={scale_x}  scale_y={scale_y}  orig={W}x{H}")

    out_path = args.out or os.path.join(args.session_dir, "gt.txt")

    out_rows: list[list[str]] = []
    n_in = 0
    n_skipped_empty = 0
    n_clamped = 0
    frames_seen: set[int] = set()
    ids_seen: set[int] = set()

    with open(args.cvat_gt, newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                n_skipped_empty += 1
                continue
            # MOT 1.1 формат: frame,id,x,y,w,h,conf,cls,vis
            # деякі експортери дають 7 полів — добиваємо дефолтами
            try:
                frame = int(float(row[0]))
                tid   = int(float(row[1]))
                x     = float(row[2]); y = float(row[3])
                w     = float(row[4]); h = float(row[5])
                conf  = float(row[6]) if len(row) > 6 else 1.0
                cls   = int(float(row[7])) if len(row) > 7 else 1
                vis   = float(row[8]) if len(row) > 8 else 1.0
            except (ValueError, IndexError) as e:
                print(f"[cvt] skip malformed row: {row}  ({e})", file=sys.stderr)
                continue
            n_in += 1

            # переходимо у систему сенсора
            x_s = x / scale_x
            y_s = y / scale_y
            w_s = w / scale_x
            h_s = h / scale_y

            if not args.no_clamp:
                x0 = max(0.0, x_s);            y0 = max(0.0, y_s)
                x1 = min(float(W), x_s + w_s); y1 = min(float(H), y_s + h_s)
                if x0 != x_s or y0 != y_s or x1 != x_s + w_s or y1 != y_s + h_s:
                    n_clamped += 1
                x_s, y_s = x0, y0
                w_s, h_s = max(0.0, x1 - x0), max(0.0, y1 - y0)

            if w_s <= 0 or h_s <= 0:
                # повністю поза кадром — пропускаємо
                continue

            if not args.no_round:
                # округляємо bbox до пікселів сенсора з мінімальною шириною/висотою 1
                xi = int(round(x_s)); yi = int(round(y_s))
                wi = max(1, int(round(w_s))); hi = max(1, int(round(h_s)))
                # ще раз клампимо після округлення
                if not args.no_clamp:
                    xi = max(0, min(W - 1, xi))
                    yi = max(0, min(H - 1, yi))
                    wi = max(1, min(W - xi, wi))
                    hi = max(1, min(H - yi, hi))
                out_rows.append([
                    str(frame), str(tid),
                    str(xi), str(yi), str(wi), str(hi),
                    f"{conf:g}", str(cls), f"{vis:g}",
                ])
            else:
                out_rows.append([
                    str(frame), str(tid),
                    f"{x_s:.3f}", f"{y_s:.3f}", f"{w_s:.3f}", f"{h_s:.3f}",
                    f"{conf:g}", str(cls), f"{vis:g}",
                ])
            frames_seen.add(frame)
            ids_seen.add(tid)

    # ── summary ─────────────────────────────────────────
    print(f"[cvt] rows in : {n_in}")
    print(f"[cvt] rows out: {len(out_rows)}")
    print(f"[cvt] frames  : {len(frames_seen)} "
          f"(min={min(frames_seen) if frames_seen else '-'}  "
          f"max={max(frames_seen) if frames_seen else '-'})")
    print(f"[cvt] track ids: {sorted(ids_seen)}")
    if n_clamped:
        print(f"[cvt] clamped to image bounds: {n_clamped}")
    if n_skipped_empty:
        print(f"[cvt] skipped empty lines: {n_skipped_empty}")
    n_expected = int(scale.get("n_frames", 0))
    if n_expected and frames_seen and max(frames_seen) > n_expected:
        print(f"[cvt] WARN: max frame {max(frames_seen)} > n_frames {n_expected}; "
              f"перевір, що CVAT-task мав той самий labeling.mp4")

    # preview
    head = out_rows[: max(0, args.preview)]
    if head:
        print("[cvt] preview:")
        print("       frame, id, x, y, w, h, conf, cls, vis")
        for row in head:
            print("       " + ",".join(row))

    if args.dry_run:
        print("[cvt] --dry-run: нічого не записано")
        return

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        # MOTChallenge не вимагає заголовка
        for row in out_rows:
            w.writerow(row)
    print(f"[cvt] wrote {out_path}")


if __name__ == "__main__":
    main()
