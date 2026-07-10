"""DeepSORT з аппірансом на основі гістограми температур патчу.

Винесено з deepsort_temp_tracker_pi_001.py.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_MAX_AGE      = 8
DEFAULT_MIN_HITS     = 3
DEFAULT_LAMBDA_COST  = 0.5      # 0 = appearance only, 1 = motion only
DEFAULT_GALLERY_SIZE = 5
DEFAULT_N_FEAT_BINS  = 16
DEFAULT_GATE_CHI2    = 9.4877   # chi2.ppf(0.95, df=4)
DEFAULT_IOU_THR      = 0.05

# Глобальний діапазон температур для нормалізації гістограми.
# Можна оновлювати ковзним середнім у real-time-режимі.
_T_MIN: float | None = None
_T_MAX: float | None = None


def set_temp_range(t_min: float, t_max: float) -> None:
    """Задати глобальний діапазон [t_min, t_max] для бінів гістограми."""
    global _T_MIN, _T_MAX
    _T_MIN = float(t_min)
    _T_MAX = float(t_max)


def get_temp_range() -> tuple[float, float]:
    if _T_MIN is None or _T_MAX is None:
        raise RuntimeError(
            "trackers.deepsort_temp: викличте set_temp_range(t_min, t_max) "
            "перед extract_temp_feature()")
    return _T_MIN, _T_MAX


def extract_temp_feature(frame2d: np.ndarray, bbox,
                         n_bins: int = DEFAULT_N_FEAT_BINS) -> np.ndarray:
    """Нормована L2-гістограма температур усередині bbox."""
    t_min, t_max = get_temp_range()
    h, w = frame2d.shape
    r0, r1, c0, c1 = bbox
    patch = frame2d[max(0, r0):min(h, r1 + 1),
                    max(0, c0):min(w, c1 + 1)].ravel()
    if patch.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    hist, _ = np.histogram(patch, bins=n_bins, range=(t_min, t_max))
    hist = hist.astype(np.float32)
    n = float(np.linalg.norm(hist))
    return hist / n if n > 1e-6 else hist


def _gallery_dist(feat, gallery) -> float:
    if not gallery:
        return 1.0
    return float(min(1.0 - float(np.dot(feat, f)) for f in gallery))


def _mah_dist(z, mean, cov) -> float:
    d = z - mean
    try:
        ch = np.linalg.cholesky(cov)
        s  = np.linalg.solve(ch, d)
        return float(np.dot(s, s))
    except np.linalg.LinAlgError:
        return 1e9


def _iou(a, b) -> float:
    r0a, r1a, c0a, c1a = a; r0b, r1b, c0b, c1b = b
    ih = max(0, min(r1a, r1b) - max(r0a, r0b))
    iw = max(0, min(c1a, c1b) - max(c0a, c0b))
    inter = ih * iw
    area_a = max(0, r1a - r0a) * max(0, c1a - c0a)
    area_b = max(0, r1b - r0b) * max(0, c1b - c0b)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class DeepSORTTempTrack:
    _id_counter = 0

    def __init__(self, bbox, feat,
                 gallery_size: int = DEFAULT_GALLERY_SIZE,
                 min_hits: int = DEFAULT_MIN_HITS):
        DeepSORTTempTrack._id_counter += 1
        self.track_id = DeepSORTTempTrack._id_counter
        r0, r1, c0, c1 = bbox
        cx, cy = (c0 + c1) / 2., (r0 + r1) / 2.
        w,  h  = float(c1 - c0 + 1), float(r1 - r0 + 1)
        self.F = np.eye(6); self.F[0, 4] = self.F[1, 5] = 1.0
        self.H = np.zeros((4, 6)); np.fill_diagonal(self.H, 1.0)
        self.Q = np.diag([.5, .5, .5, .5, 1., 1.])
        self.R = np.diag([1., 1., 2., 2.])
        self.P = np.diag([2., 2., 4., 4., 10., 10.])
        self.x = np.array([cx, cy, w, h, 0., 0.])
        self.gallery: list[np.ndarray] = [feat] if feat is not None else []
        self.hits = 1
        self.age  = 1
        self.missed = 0
        self.confirmed = False
        self._gallery_size = gallery_size
        self._min_hits = min_hits

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[2] = max(self.x[2], 1.); self.x[3] = max(self.x[3], 1.)
        self.age += 1; self.missed += 1

    def update(self, bbox, feat):
        r0, r1, c0, c1 = bbox
        z = np.array([(c0 + c1) / 2., (r0 + r1) / 2.,
                       float(c1 - c0 + 1), float(r1 - r0 + 1)])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.x[2] = max(self.x[2], 1.); self.x[3] = max(self.x[3], 1.)
        if feat is not None:
            self.gallery.append(feat)
            if len(self.gallery) > self._gallery_size:
                self.gallery.pop(0)
        self.missed = 0; self.hits += 1
        if self.hits >= self._min_hits:
            self.confirmed = True

    def get_bbox(self):
        cx, cy, w, h = self.x[:4]
        return (round(cy - h / 2), round(cy + h / 2),
                round(cx - w / 2), round(cx + w / 2))

    def get_velocity(self):
        return float(self.x[4]), float(self.x[5])

    def get_measurement_mean(self):
        return self.H @ self.x

    def get_innovation_covariance(self):
        return self.H @ self.P @ self.H.T + self.R


class DeepSORTTempTracker:
    """DeepSORT з temperature-histogram ReID. Уніфікований API."""

    def __init__(self,
                 max_age: int = DEFAULT_MAX_AGE,
                 min_hits: int = DEFAULT_MIN_HITS,
                 lambda_cost: float = DEFAULT_LAMBDA_COST,
                 gallery_size: int = DEFAULT_GALLERY_SIZE,
                 n_feat_bins: int = DEFAULT_N_FEAT_BINS,
                 gate_chi2: float = DEFAULT_GATE_CHI2,
                 iou_thr: float = DEFAULT_IOU_THR):
        self.tracks: list[DeepSORTTempTrack] = []
        self._max_age      = max_age
        self._min_hits     = min_hits
        self._lambda_cost  = lambda_cost
        self._gallery_size = gallery_size
        self._n_feat_bins  = n_feat_bins
        self._gate_chi2    = gate_chi2
        self._iou_thr      = iou_thr

    def reset(self):
        self.tracks = []
        DeepSORTTempTrack._id_counter = 0

    def predict_all(self):
        for t in self.tracks: t.predict()

    def _cost(self, dets, feats, subset):
        N, M = len(dets), len(subset)
        C = np.zeros((N, M))
        for i, (db, df) in enumerate(zip(dets, feats)):
            r0, r1, c0, c1 = db
            z = np.array([(c0 + c1) / 2., (r0 + r1) / 2.,
                           float(c1 - c0 + 1), float(r1 - r0 + 1)])
            for j, t in enumerate(subset):
                if _mah_dist(z, t.get_measurement_mean(),
                             t.get_innovation_covariance()) > self._gate_chi2:
                    C[i, j] = 1e9
                    continue
                cm = 1.0 - _iou(db, t.get_bbox())
                ca = _gallery_dist(df, t.gallery)
                C[i, j] = self._lambda_cost * cm + (1 - self._lambda_cost) * ca
        return C

    def _assign(self, C, thr=0.9):
        if C.size == 0:
            return [], list(range(C.shape[0])), list(range(C.shape[1]))
        ri, ci = linear_sum_assignment(C)
        matched, ud, ut = [], [], []
        for r, c in zip(ri, ci):
            if C[r, c] < thr: matched.append((r, c))
            else: ud.append(r); ut.append(c)
        ud += [i for i in range(C.shape[0]) if i not in ri]
        ut += [j for j in range(C.shape[1]) if j not in ci]
        return matched, ud, ut

    def update(self, detections, frame2d):
        feats = [extract_temp_feature(frame2d, b, self._n_feat_bins)
                 for b in detections]
        unm_d = list(range(len(detections)))
        unm_t = list(range(len(self.tracks)))

        for level in range(1, self._max_age + 1):
            if not unm_d or not unm_t: break
            li = [j for j in unm_t if self.tracks[j].missed == level]
            if not li: continue
            C = self._cost([detections[i] for i in unm_d],
                           [feats[i]       for i in unm_d],
                           [self.tracks[j] for j in li])
            matched, _, _ = self._assign(C)
            for dl, tl in matched:
                dg = unm_d[dl]; tg = li[tl]
                self.tracks[tg].update(detections[dg], feats[dg])
                unm_d.remove(dg); unm_t.remove(tg)

        if unm_d and unm_t:
            rd = [detections[i] for i in unm_d]
            rt = [self.tracks[j] for j in unm_t]
            IM = np.array([[_iou(d, t.get_bbox()) for t in rt] for d in rd],
                          dtype=np.float32)
            if IM.size > 0:
                ri, ci = linear_sum_assignment(1 - IM)
                done_d, done_t = [], []
                for r, c in zip(ri, ci):
                    if IM[r, c] >= self._iou_thr:
                        self.tracks[unm_t[c]].update(
                            detections[unm_d[r]], feats[unm_d[r]])
                        done_d.append(unm_d[r]); done_t.append(unm_t[c])
                unm_d = [i for i in unm_d if i not in done_d]

        for i in unm_d:
            self.tracks.append(DeepSORTTempTrack(
                detections[i], feats[i],
                gallery_size=self._gallery_size,
                min_hits=self._min_hits))
        self.tracks = [t for t in self.tracks if t.missed <= self._max_age]

    def get_active(self):
        return [t for t in self.tracks if t.confirmed]

    get_active_tracks = get_active

    def get_best(self):
        pool = self.get_active() or self.tracks
        return max(pool, key=lambda t: t.hits) if pool else None

    get_best_track = get_best
