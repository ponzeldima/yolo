"""DeepSORT з CNN-аппірансом (MicroReIDIR).

Винесено з deepsort_cnn_tracker_pi_002.py. Імпорт torch робиться ліниво,
щоб модуль можна було імпортувати навіть без встановленого torch
(наприклад, для запуску SORT/Temp у тому ж процесі без CNN).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import linear_sum_assignment

# --- параметри CNN-аппірансу ---
DEFAULT_PATCH_H   = 16
DEFAULT_PATCH_W   = 16
DEFAULT_EMBED_DIM = 16

# --- DeepSORT-параметри ---
DEFAULT_MAX_AGE      = 10
DEFAULT_MIN_HITS     = 2
DEFAULT_LAMBDA_COST  = 0.5
DEFAULT_GALLERY_SIZE = 5
DEFAULT_GATE_CHI2    = 9.4877
DEFAULT_IOU_THR      = 0.05

DEVICE = "cpu"


# ─── CNN ────────────────────────────────────────────────────
def _torch_modules():
    """Лінивий імпорт torch — щоб модуль вантажився без torch."""
    import torch                                # noqa: WPS433
    import torch.nn as nn                       # noqa: WPS433
    import torch.nn.functional as F             # noqa: WPS433
    return torch, nn, F


class MicroReIDIR:
    """Wrapper над torch.nn.Module — створюється тільки при load_model().

    Сама архітектура зберігається ідентично оригіналу.
    """
    def __new__(cls, embed_dim: int = DEFAULT_EMBED_DIM, dropout: float = 0.3):
        torch, nn, F = _torch_modules()

        class ConvBNReLU(nn.Module):
            def __init__(self, in_ch, out_ch, kernel=3, padding=1):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            def forward(self, x):  # noqa: D401
                return self.block(x)

        class _Model(nn.Module):
            def __init__(self, embed_dim, dropout):
                super().__init__()
                self.block1 = nn.Sequential(
                    ConvBNReLU(1, 16), ConvBNReLU(16, 16), nn.MaxPool2d(2, 2))
                self.block2 = nn.Sequential(
                    ConvBNReLU(16, 32), ConvBNReLU(32, 32), nn.MaxPool2d(2, 2))
                self.block3 = nn.Sequential(
                    ConvBNReLU(32, 64), nn.AdaptiveAvgPool2d(1))
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64, 32, bias=False),
                    nn.BatchNorm1d(32),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(32, embed_dim),
                )
            def forward(self, x):
                x = self.block1(x)
                x = self.block2(x)
                x = self.block3(x)
                x = self.head(x)
                return F.normalize(x, p=2, dim=1)

        return _Model(embed_dim, dropout)


def load_model(path: str, device: str = DEVICE):
    """Завантажити чекпойнт MicroReIDIR.

    Повертає (model_eval, embed_dim).
    """
    torch, _, _ = _torch_modules()
    checkpoint  = torch.load(path, map_location=device, weights_only=False)
    embed_dim   = checkpoint.get("embed_dim", DEFAULT_EMBED_DIM)
    model       = MicroReIDIR(embed_dim=embed_dim)
    model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, int(embed_dim)


# ─── фічі ────────────────────────────────────────────────
def _crop_patch(frame2d: np.ndarray, bbox,
                patch_h: int, patch_w: int,
                t_min: float, t_max: float):
    r0, r1, c0, c1 = bbox
    h, w = frame2d.shape
    r0c = max(0, r0); r1c = min(h, r1 + 1)
    c0c = max(0, c0); c1c = min(w, c1 + 1)
    patch = frame2d[r0c:r1c, c0c:c1c]
    ph, pw = patch.shape
    if ph < 2 or pw < 2:
        return None
    patch_resized = zoom(patch, (patch_h / ph, patch_w / pw),
                         order=1).astype(np.float32)
    patch_norm = np.clip((patch_resized - t_min) / (t_max - t_min + 1e-8),
                         0.0, 1.0)
    return patch_norm


def extract_cnn_feature(model, frame2d: np.ndarray, bbox,
                        t_min: float, t_max: float,
                        embed_dim: int,
                        patch_h: int = DEFAULT_PATCH_H,
                        patch_w: int = DEFAULT_PATCH_W,
                        device: str = DEVICE) -> np.ndarray:
    torch, _, _ = _torch_modules()
    patch = _crop_patch(frame2d, bbox, patch_h, patch_w, t_min, t_max)
    if patch is None:
        return np.zeros(embed_dim, dtype=np.float32)
    tensor = torch.tensor(patch, dtype=torch.float32) \
        .unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor)
    return embedding.cpu().numpy()[0]


# ─── метрики ────────────────────────────────────────────
def _cosine_distance(a, b) -> float:
    return 1.0 - float(np.dot(a, b))


def _gallery_dist(feat, gallery) -> float:
    if not gallery:
        return 1.0
    return float(min(_cosine_distance(feat, f) for f in gallery))


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


# ─── Track / Manager ────────────────────────────────────
class DeepSORTCNNTrack:
    _id_counter = 0

    def __init__(self, bbox, feature,
                 gallery_size: int = DEFAULT_GALLERY_SIZE,
                 min_hits: int = DEFAULT_MIN_HITS):
        DeepSORTCNNTrack._id_counter += 1
        self.track_id = DeepSORTCNNTrack._id_counter
        r0, r1, c0, c1 = bbox
        cx = (c0 + c1) / 2.;  cy = (r0 + r1) / 2.
        w  = float(c1 - c0 + 1); h = float(r1 - r0 + 1)
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 4] = self.F[1, 5] = 1.0
        self.H = np.zeros((4, 6), dtype=np.float64)
        np.fill_diagonal(self.H, 1.0)
        self.Q = np.diag([0.5, 0.5, 0.5, 0.5, 1.0, 1.0]).astype(np.float64)
        self.R = np.diag([1.0, 1.0, 2.0, 2.0]).astype(np.float64)
        self.P = np.diag([2.0, 2.0, 4.0, 4.0, 10.0, 10.0]).astype(np.float64)
        self.x = np.array([cx, cy, w, h, 0., 0.], dtype=np.float64)

        self.gallery   = [feature] if feature is not None else []
        self.hits      = 1
        self.age       = 1
        self.missed    = 0
        self.confirmed = False
        self._gallery_size = gallery_size
        self._min_hits = min_hits

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[2] = max(self.x[2], 1.); self.x[3] = max(self.x[3], 1.)
        self.age += 1; self.missed += 1

    def update(self, bbox, feature):
        r0, r1, c0, c1 = bbox
        z = np.array([(c0 + c1) / 2., (r0 + r1) / 2.,
                       float(c1 - c0 + 1), float(r1 - r0 + 1)])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.x[2] = max(self.x[2], 1.); self.x[3] = max(self.x[3], 1.)
        if feature is not None:
            self.gallery.append(feature)
            if len(self.gallery) > self._gallery_size:
                self.gallery.pop(0)
        self.missed = 0; self.hits += 1
        if self.hits >= self._min_hits:
            self.confirmed = True

    def get_bbox(self):
        cx, cy, w, h = self.x[:4]
        return (int(round(cy - h / 2)), int(round(cy + h / 2)),
                int(round(cx - w / 2)), int(round(cx + w / 2)))

    def get_velocity(self):
        return float(self.x[4]), float(self.x[5])

    def get_measurement_mean(self):
        return self.H @ self.x

    def get_innovation_covariance(self):
        return self.H @ self.P @ self.H.T + self.R


class DeepSORTCNNTracker:
    """DeepSORT з CNN-ReID. Уніфікований API.

    Параметри t_min / t_max використовуються для нормалізації патчу перед
    подачею в CNN. Якщо у real-time-режимі шкала повзе, оновлюйте їх
    методом set_temp_range().
    """

    def __init__(self, model, embed_dim: int,
                 t_min: float, t_max: float,
                 max_age: int = DEFAULT_MAX_AGE,
                 min_hits: int = DEFAULT_MIN_HITS,
                 lambda_cost: float = DEFAULT_LAMBDA_COST,
                 gallery_size: int = DEFAULT_GALLERY_SIZE,
                 gate_chi2: float = DEFAULT_GATE_CHI2,
                 iou_thr: float = DEFAULT_IOU_THR,
                 patch_h: int = DEFAULT_PATCH_H,
                 patch_w: int = DEFAULT_PATCH_W,
                 device: str = DEVICE):
        self.tracks: list[DeepSORTCNNTrack] = []
        self.model         = model
        self.embed_dim     = embed_dim
        self._t_min        = t_min
        self._t_max        = t_max
        self._max_age      = max_age
        self._min_hits     = min_hits
        self._lambda_cost  = lambda_cost
        self._gallery_size = gallery_size
        self._gate_chi2    = gate_chi2
        self._iou_thr      = iou_thr
        self._patch_h      = patch_h
        self._patch_w      = patch_w
        self._device       = device

    def set_temp_range(self, t_min: float, t_max: float) -> None:
        self._t_min = float(t_min); self._t_max = float(t_max)

    def reset(self):
        self.tracks = []
        DeepSORTCNNTrack._id_counter = 0

    def predict_all(self):
        for t in self.tracks: t.predict()

    def _feature(self, frame2d, bbox):
        return extract_cnn_feature(self.model, frame2d, bbox,
                                   self._t_min, self._t_max,
                                   self.embed_dim,
                                   self._patch_h, self._patch_w,
                                   self._device)

    def _cost(self, dets, feats, subset):
        N, M = len(dets), len(subset)
        C = np.zeros((N, M), dtype=np.float64)
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
        feats = [self._feature(frame2d, b) for b in detections]
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
            self.tracks.append(DeepSORTCNNTrack(
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
