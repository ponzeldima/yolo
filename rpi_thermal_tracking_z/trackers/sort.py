"""Класичний SORT: Kalman (constant-velocity) + Hungarian по IoU.

Винесено з SORT_tracker_realtime.py — щоб ту саму логіку можна було
використати у реальному часі та у скриптах прогону записаних сесій.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_IOU_THRESHOLD = 0.07
DEFAULT_MAX_AGE       = 7
DEFAULT_MIN_HITS      = 3


def _iou_matrix(detections, predicted) -> np.ndarray:
    if not detections or not predicted:
        return np.zeros((len(detections), len(predicted)), dtype=np.float32)
    D = np.array(detections, dtype=np.float32)
    P = np.array(predicted,  dtype=np.float32)
    ih = np.maximum(0, np.minimum(D[:, 1:2], P[:, 1]) - np.maximum(D[:, 0:1], P[:, 0]))
    iw = np.maximum(0, np.minimum(D[:, 3:4], P[:, 3]) - np.maximum(D[:, 2:3], P[:, 2]))
    inter  = ih * iw
    area_d = np.maximum(0, D[:, 1] - D[:, 0]) * np.maximum(0, D[:, 3] - D[:, 2])
    area_p = np.maximum(0, P[:, 1] - P[:, 0]) * np.maximum(0, P[:, 3] - P[:, 2])
    union  = area_d[:, None] + area_p[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _hungarian(iou_mat, threshold):
    if iou_mat.size == 0:
        return [], list(range(iou_mat.shape[0])), list(range(iou_mat.shape[1]))
    row_ind, col_ind = linear_sum_assignment(1.0 - iou_mat)
    matched, unm_d, unm_t = [], [], []
    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= threshold:
            matched.append((r, c))
        else:
            unm_d.append(r); unm_t.append(c)
    for i in range(iou_mat.shape[0]):
        if i not in row_ind: unm_d.append(i)
    for j in range(iou_mat.shape[1]):
        if j not in col_ind: unm_t.append(j)
    return matched, unm_d, unm_t


class SORTTrack:
    _id_counter = 0

    def __init__(self, bbox, min_hits: int = DEFAULT_MIN_HITS):
        SORTTrack._id_counter += 1
        self.track_id = SORTTrack._id_counter
        r0, r1, c0, c1 = bbox
        cx, cy = (c0 + c1) / 2.0, (r0 + r1) / 2.0
        w,  h  = float(c1 - c0 + 1), float(r1 - r0 + 1)
        self.F = np.eye(6, dtype=np.float64); self.F[0, 4] = self.F[1, 5] = 1.0
        self.H = np.zeros((4, 6), dtype=np.float64); np.fill_diagonal(self.H, 1.0)
        self.Q = np.diag([0.5, 0.5, 0.5, 0.5, 1.0, 1.0]).astype(np.float64)
        self.R = np.diag([1.0, 1.0, 2.0, 2.0]).astype(np.float64)
        self.P = np.diag([2.0, 2.0, 4.0, 4.0, 10.0, 10.0]).astype(np.float64)
        self.x = np.array([cx, cy, w, h, 0., 0.], dtype=np.float64)
        self.hits = 1
        self.age  = 1
        self.missed = 0
        self.confirmed = False
        self._min_hits = min_hits
        self.gallery = []  # для уніфікованого API з DeepSORT-трекерами

    def predict(self):
        self.x   = self.F @ self.x
        self.P   = self.F @ self.P @ self.F.T + self.Q
        self.x[2] = max(self.x[2], 1.0); self.x[3] = max(self.x[3], 1.0)
        self.age += 1; self.missed += 1

    def update(self, bbox):
        r0, r1, c0, c1 = bbox
        z = np.array([(c0 + c1) / 2., (r0 + r1) / 2.,
                       float(c1 - c0 + 1), float(r1 - r0 + 1)])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.x[2] = max(self.x[2], 1.0); self.x[3] = max(self.x[3], 1.0)
        self.missed = 0; self.hits += 1
        if self.hits >= self._min_hits:
            self.confirmed = True

    def get_bbox(self):
        cx, cy, w, h = self.x[:4]
        return (round(cy - h / 2), round(cy + h / 2),
                round(cx - w / 2), round(cx + w / 2))

    def get_velocity(self):
        return float(self.x[4]), float(self.x[5])


class SORTTracker:
    """Менеджер SORT-треків.

    Уніфікований API:
        predict_all()
        update(detections, frame2d=None)   # frame2d ігнорується (для API-сумісності)
        reset()
        get_active() / get_best()
        tracks : list[SORTTrack]
    """

    def __init__(self,
                 iou_threshold: float = DEFAULT_IOU_THRESHOLD,
                 max_age: int = DEFAULT_MAX_AGE,
                 min_hits: int = DEFAULT_MIN_HITS):
        self.tracks: list[SORTTrack] = []
        self._iou_thr  = iou_threshold
        self._max_age  = max_age
        self._min_hits = min_hits

    def reset(self):
        self.tracks = []
        SORTTrack._id_counter = 0

    def predict_all(self):
        for t in self.tracks: t.predict()

    def update(self, detections, frame2d=None):
        predicted = [t.get_bbox() for t in self.tracks]
        if self.tracks and detections:
            iou_mat = _iou_matrix(detections, predicted)
            matched, unm_d, unm_t = _hungarian(iou_mat, self._iou_thr)
            for d_idx, t_idx in matched:
                self.tracks[t_idx].update(detections[d_idx])
            for t_idx in unm_t:
                self.tracks[t_idx].missed += 1
        else:
            unm_d = list(range(len(detections)))
        for d_idx in unm_d:
            self.tracks.append(SORTTrack(detections[d_idx], min_hits=self._min_hits))
        self.tracks = [t for t in self.tracks if t.missed <= self._max_age]

    def get_active(self):
        return [t for t in self.tracks if t.confirmed]

    # Aliases for compatibility with older code
    get_active_tracks = get_active

    def get_best(self):
        active = self.get_active()
        pool   = active if active else self.tracks
        return max(pool, key=lambda t: t.hits) if pool else None

    get_best_track = get_best
