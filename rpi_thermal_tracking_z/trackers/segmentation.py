"""MRF + Graph-Cut сегментація теплового кадру.

Об'єднує три майже ідентичні реалізації з:
  SORT_tracker_realtime.segment_frame
  deepsort_temp_tracker_pi_001.segment_frame
  deepsort_cnn_tracker_pi_002.segment_frame

Параметри (lam, k_sigma) можна перевизначати — це дозволяє кожному
трекеру лишитися зі своїми дефолтами.
"""
from __future__ import annotations

import numpy as np
import maxflow
from scipy.ndimage import label as scipy_label

FRAME_H = 24
FRAME_W = 32

DEFAULT_LAMBDA  = 1.5    # вага регуляризації MRF
DEFAULT_K_SIGMA = 2.5    # поріг: mean + K_SIGMA * std

_NODE_IDS = np.arange(FRAME_H * FRAME_W, dtype=np.int32).reshape(FRAME_H, FRAME_W)


def segment_frame(frame2d: np.ndarray,
                  k_sigma: float = DEFAULT_K_SIGMA,
                  lam: float = DEFAULT_LAMBDA):
    """Сегментувати кадр на 2 класи (фон/об'єкт).

    Повертає (mask, bbox) або (None, None), де
        mask : np.bool_ (H, W)
        bbox : (r0, r1, c0, c1) у координатах сенсора.
    """
    flat   = frame2d.ravel()
    thresh = flat.mean() + k_sigma * flat.std()
    mask_bg = flat < thresh
    if mask_bg.all() or mask_bg.sum() < 2:
        return None, None

    fg, bg = flat[~mask_bg], flat[mask_bg]
    mu  = [bg.mean(), fg.mean()]
    sig = [max(bg.std(), 0.5), max(fg.std(), 0.5)]

    h, w = frame2d.shape
    unary0 = (np.log(sig[0] * np.sqrt(2 * np.pi)) +
              (flat - mu[0]) ** 2 / (2 * sig[0] ** 2)).reshape(h, w)
    unary1 = (np.log(sig[1] * np.sqrt(2 * np.pi)) +
              (flat - mu[1]) ** 2 / (2 * sig[1] ** 2)).reshape(h, w)

    if h == FRAME_H and w == FRAME_W:
        node_ids = _NODE_IDS
    else:
        node_ids = np.arange(h * w, dtype=np.int32).reshape(h, w)

    g = maxflow.Graph[float](h * w, 0)
    g.add_nodes(h * w)
    g.add_grid_tedges(node_ids, unary1, unary0)
    g.add_grid_edges(node_ids, lam)
    g.maxflow()

    labels  = g.get_grid_segments(node_ids).astype(np.int32)
    lbl_map, n = scipy_label(labels, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return None, None

    sizes = [int((lbl_map == i).sum()) for i in range(1, n + 1)]
    mask  = (lbl_map == (int(np.argmax(sizes)) + 1))

    rows_hit = np.any(mask, axis=1)
    cols_hit = np.any(mask, axis=0)
    r0 = int(np.where(rows_hit)[0][0]); r1 = int(np.where(rows_hit)[0][-1])
    c0 = int(np.where(cols_hit)[0][0]); c1 = int(np.where(cols_hit)[0][-1])
    return mask, (r0, r1, c0, c1)
