#!/usr/bin/env python3
"""
Прогнати трекери з пакета `trackers/` на записаній сесії, порівняти з GT
(MOTChallenge-формат у координатах сенсора 24×32) і надрукувати метрики
(MOTA, MOTP, IDF1, ID switches, precision/recall, IoU/center-error по кадрах).

Приклад:
    python3 rpi_thermal_tracking_z/evaluate_trackers_on_session.py sessions/session_5

Опції:
    --gt sessions/session_5/gt.txt   — інший шлях до GT (за замовч. <session_dir>/gt.txt)
    --iou-thr 0.5                     — поріг IoU для зіставлення (motmetrics)
    --no-cnn                          — пропустити CNN-трекер
    --per-frame-csv out.csv           — зберегти per-frame IoU/center-error
    --plot iou.png                    — зберегти графік IoU(t) для всіх трекерів
    --hyp-dir runs/eval               — зберегти hypothesis MOT-файли (для TrackEval)

Залежності:
    pip install motmetrics  (для метрик; matplotlib — опційно для --plot)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any

import numpy as np

# motmetrics 1.4 використовує np.asfarray, який видалили в NumPy 2.0.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trackers.segmentation import (  # noqa: E402
    segment_frame, DEFAULT_K_SIGMA, DEFAULT_LAMBDA,
)
from trackers.sort import SORTTrack, SORTTracker  # noqa: E402
from trackers.deepsort_temp import (  # noqa: E402
    DeepSORTTempTrack, DeepSORTTempTracker, set_temp_range,
)

try:
    import motmetrics as mm
except ImportError:
    print("ERROR: motmetrics не встановлений. Запусти:\n"
          "  .venv/bin/pip install motmetrics", file=sys.stderr)
    sys.exit(1)


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
    shape = tuple(meta["shape"])
    frames = np.fromfile(f32_path, dtype="float32").reshape(-1, *shape)
    ts     = np.fromfile(ts_path,  dtype="float64")
    n = min(len(frames), len(ts))
    return frames[:n], ts[:n], meta


# ── GT loader ─────────────────────────────────────────────
def load_gt(path: str) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """frame_idx (0-based) → list of (id, x, y, w, h) у системі сенсора.

    MOTChallenge використовує 1-based frame index — переводимо в 0-based.
    """
    gt: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#"):
                continue
            try:
                frame = int(float(row[0])) - 1   # → 0-based
                tid   = int(float(row[1]))
                x     = float(row[2]); y = float(row[3])
                w     = float(row[4]); h = float(row[5])
                conf  = float(row[6]) if len(row) > 6 else 1.0
            except (ValueError, IndexError):
                continue
            # MOT GT іноді має conf=0 для "ignore"
            if conf <= 0:
                continue
            gt[frame].append((tid, x, y, w, h))
    return gt


# ── bbox helpers ──────────────────────────────────────────
def _rrcc_to_xywh(bbox: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    """(r0, r1, c0, c1) → (x, y, w, h). Включно по r/c, так само як трекер."""
    r0, r1, c0, c1 = bbox
    return float(c0), float(r0), float(c1 - c0 + 1), float(r1 - r0 + 1)


def _bbox_center(b: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = b
    return x + w / 2.0, y + h / 2.0


# ── pretty print ─────────────────────────────────────────
def _print_summary(name: str, acc: mm.MOTAccumulator,
                   per_frame: list[dict[str, Any]]) -> None:
    mh = mm.metrics.create()
    metrics = [
        "num_frames",
        "mota", "motp", "idf1",
        "precision", "recall",
        "num_switches", "num_fragmentations",
        "num_false_positives", "num_misses",
        "mostly_tracked", "mostly_lost",
        "num_unique_objects",
    ]
    summary = mh.compute(acc, metrics=metrics, name=name)
    table = mm.io.render_summary(summary, formatters=mh.formatters,
                                 namemap=mm.io.motchallenge_metric_names)
    print(table)

    # додаткова per-frame статистика
    ious = [r["iou"] for r in per_frame if r["iou"] is not None]
    cers = [r["center_err"] for r in per_frame if r["center_err"] is not None]
    n_match = len(ious)
    n_total = len(per_frame)
    print(f"  per-frame: matched {n_match}/{n_total}  "
          f"mean IoU={np.mean(ious):.3f}  med IoU={np.median(ious):.3f}  "
          f"mean center-err={np.mean(cers):.2f}px  "
          f"med center-err={np.median(cers):.2f}px"
          if n_match else "  per-frame: no GT-tracker overlaps")
    print()


def _write_hyp_mot(path: str, hyp_rows: list[tuple[int, int, float, float, float, float]]) -> None:
    """Записати hypothesis у MOT-формат (1-based frame) для TrackEval/візуалізатора."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        for frame0, tid, x, y, bw, bh in hyp_rows:
            w.writerow([frame0 + 1, tid, f"{x:.3f}", f"{y:.3f}",
                        f"{bw:.3f}", f"{bh:.3f}", "1", "1", "1"])


# ── core eval ────────────────────────────────────────────
def evaluate_one(name: str, tracker, frames: np.ndarray,
                 gt: dict, k_sigma: float, lam: float,
                 needs_frame: bool, iou_thr: float
                 ) -> tuple[mm.MOTAccumulator, list[dict], list[tuple]]:
    """Прогнати трекер і повернути (accumulator, per_frame, hyp_rows)."""
    acc = mm.MOTAccumulator(auto_id=True)
    per_frame: list[dict[str, Any]] = []
    hyp_rows: list[tuple] = []

    for fidx in range(len(frames)):
        frame2d = frames[fidx]
        _mask, bbox_mrf = segment_frame(frame2d, k_sigma=k_sigma, lam=lam)
        detections = [bbox_mrf] if bbox_mrf is not None else []

        tracker.predict_all()
        if needs_frame:
            tracker.update(detections, frame2d)
        else:
            tracker.update(detections)

        # GT bboxes для цього кадру
        gt_items = gt.get(fidx, [])
        gt_ids  = [item[0] for item in gt_items]
        gt_xywh = [item[1:] for item in gt_items]

        # hypothesis: усі активні треки (включно з невизначеними — щоб не "приховати"
        # ранні передбачення; іншому трекеру при `confirmed_only` буде нечесно)
        hyp_ids = []
        hyp_xywh = []
        for t in tracker.tracks:
            xywh = _rrcc_to_xywh(t.get_bbox())
            hyp_ids.append(int(t.track_id))
            hyp_xywh.append(xywh)
            hyp_rows.append((fidx, int(t.track_id), *xywh))

        # IoU-distance matrix для motmetrics
        dists = mm.distances.iou_matrix(gt_xywh, hyp_xywh, max_iou=1.0 - iou_thr) \
            if gt_xywh and hyp_xywh else np.full((len(gt_xywh), len(hyp_xywh)), np.nan)
        acc.update(gt_ids, hyp_ids, dists)

        # per-frame метрика: беремо найкраще зіставлення GT_id=1 ↔ best-track
        best_iou = None
        best_cer = None
        if gt_xywh and hyp_xywh:
            # IoU matrix без max_iou-clamp'у для self-діагностики
            ious_full = 1.0 - mm.distances.iou_matrix(gt_xywh, hyp_xywh, max_iou=1.0)
            ious_full = np.nan_to_num(ious_full, nan=0.0)
            i = 0  # перший GT (у нас зазвичай одна кружка)
            j = int(np.argmax(ious_full[i]))
            best_iou = float(ious_full[i, j])
            gx, gy = _bbox_center(gt_xywh[i])
            hx, hy = _bbox_center(hyp_xywh[j])
            best_cer = float(np.hypot(gx - hx, gy - hy))
        per_frame.append({"frame": fidx, "iou": best_iou, "center_err": best_cer,
                          "n_gt": len(gt_xywh), "n_hyp": len(hyp_xywh)})
    return acc, per_frame, hyp_rows


# ── main ─────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Evaluate trackers vs CVAT-розмічений GT на сесії MLX90642")
    p.add_argument("session_dir")
    p.add_argument("--gt", default=None,
                   help="шлях до gt.txt (за замовч. <session_dir>/gt.txt)")
    p.add_argument("--iou-thr", type=float, default=0.5,
                   help="поріг IoU для зіставлення (motmetrics)")
    p.add_argument("--k-sigma", type=float, default=DEFAULT_K_SIGMA)
    p.add_argument("--lam",     type=float, default=DEFAULT_LAMBDA)
    p.add_argument("--no-cnn", action="store_true")
    p.add_argument("--model",
                   default=os.path.join(
                       os.path.dirname(os.path.abspath(__file__)),
                       "micro_reid_ir.pt"))
    p.add_argument("--per-frame-csv", default=None,
                   help="зберегти per-frame IoU/center-error в CSV")
    p.add_argument("--plot", default=None,
                   help="PNG з IoU(t) для всіх трекерів (потрібен matplotlib)")
    p.add_argument("--hyp-dir", default=None,
                   help="куди скласти hypothesis MOT-файли (<name>.txt)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    gt_path = args.gt or os.path.join(args.session_dir, "gt.txt")
    if not os.path.isfile(gt_path):
        print(f"GT не знайдено: {gt_path}\n"
              "  Запусти cvat_prepare.py → розмітити → cvat_to_gt.py", file=sys.stderr)
        sys.exit(1)

    frames, ts, meta = load_session(args.session_dir)
    H, W = frames.shape[1:]
    gt = load_gt(gt_path)
    n_gt_frames = len(gt)
    print(f"[eval] session={args.session_dir}  frames={len(frames)}  shape=({H},{W})")
    print(f"[eval] gt={gt_path}  frames_with_gt={n_gt_frames}  "
          f"track_ids={sorted({i for items in gt.values() for i,*_ in items})}")
    print(f"[eval] iou_thr={args.iou_thr}  k_sigma={args.k_sigma}  lam={args.lam}")
    print()

    # CNN-діапазон температур: однаковий для DeepSORT-temp (бінінг) і CNN-патчу
    t_min = float(meta.get("t_min_global_c") or frames.min())
    t_max = float(meta.get("t_max_global_c") or frames.max())
    set_temp_range(t_min, t_max)

    # ── будуємо список трекерів ─────────────────────────
    SORTTrack._id_counter = 0
    DeepSORTTempTrack._id_counter = 0
    tasks: list[tuple[str, object, bool]] = [
        ("SORT",          SORTTracker(),         False),
        ("DeepSORT-temp", DeepSORTTempTracker(), True),
    ]
    if not args.no_cnn:
        try:
            from trackers.deepsort_cnn import (  # noqa: WPS433
                DeepSORTCNNTrack, DeepSORTCNNTracker, load_model)
            if not os.path.isfile(args.model):
                print(f"[eval] WARN: model not found: {args.model} — CNN-трекер вимкнено")
            else:
                DeepSORTCNNTrack._id_counter = 0
                model, edim = load_model(args.model)
                tasks.append(("DeepSORT-CNN",
                              DeepSORTCNNTracker(model, edim, t_min, t_max),
                              True))
                print(f"[eval] CNN model loaded: {args.model} embed_dim={edim}")
        except ImportError as e:
            print(f"[eval] CNN-трекер пропущено: {e}")

    # ── евалюація ───────────────────────────────────────
    results: list[tuple[str, mm.MOTAccumulator, list[dict]]] = []
    all_hyp: dict[str, list[tuple]] = {}
    for name, tracker, needs_frame in tasks:
        acc, pf, hyp_rows = evaluate_one(
            name, tracker, frames, gt,
            k_sigma=args.k_sigma, lam=args.lam,
            needs_frame=needs_frame, iou_thr=args.iou_thr)
        _print_summary(name, acc, pf)
        results.append((name, acc, pf))
        all_hyp[name] = hyp_rows

    # ── зведена таблиця ─────────────────────────────────
    mh = mm.metrics.create()
    summary = mh.compute_many(
        [acc for _n, acc, _pf in results],
        metrics=["num_frames", "mota", "motp", "idf1",
                 "precision", "recall",
                 "num_switches", "num_fragmentations",
                 "num_false_positives", "num_misses",
                 "mostly_tracked", "mostly_lost"],
        names=[n for n, _a, _pf in results],
        generate_overall=False)
    table = mm.io.render_summary(summary, formatters=mh.formatters,
                                 namemap=mm.io.motchallenge_metric_names)
    print("──── Сумарна таблиця ────────────────────────────")
    print(table)

    # ── опційні артефакти ──────────────────────────────
    if args.per_frame_csv:
        with open(args.per_frame_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            header = ["frame"]
            for n, _a, _pf in results:
                header += [f"{n}__iou", f"{n}__center_err",
                           f"{n}__n_hyp"]
            header.append("n_gt")
            w.writerow(header)
            n_frames = max(len(pf) for _n, _a, pf in results)
            for i in range(n_frames):
                row: list[Any] = [i]
                n_gt_i = 0
                for _n, _a, pf in results:
                    rec = pf[i]
                    row += [f"{rec['iou']:.4f}" if rec["iou"] is not None else "",
                            f"{rec['center_err']:.3f}" if rec["center_err"] is not None else "",
                            rec["n_hyp"]]
                    n_gt_i = rec["n_gt"]
                row.append(n_gt_i)
                w.writerow(row)
        print(f"[eval] wrote per-frame CSV: {args.per_frame_csv}")

    if args.hyp_dir:
        os.makedirs(args.hyp_dir, exist_ok=True)
        for name, rows in all_hyp.items():
            safe = name.replace(" ", "_")
            out = os.path.join(args.hyp_dir, f"{safe}.txt")
            _write_hyp_mot(out, rows)
            print(f"[eval] wrote hypothesis MOT: {out}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[eval] matplotlib не встановлено, --plot пропущено")
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
            for name, _a, pf in results:
                xs = [r["frame"] for r in pf]
                ys = [r["iou"] if r["iou"] is not None else np.nan for r in pf]
                ax.plot(xs, ys, label=name, linewidth=1.4)
            ax.axhline(args.iou_thr, color="grey", ls="--", lw=0.8,
                       label=f"IoU thr {args.iou_thr}")
            ax.set_xlabel("frame"); ax.set_ylabel("IoU vs GT (best track)")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"IoU(t)  —  {os.path.basename(args.session_dir)}")
            ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(args.plot, dpi=120)
            print(f"[eval] wrote plot: {args.plot}")


if __name__ == "__main__":
    main()
