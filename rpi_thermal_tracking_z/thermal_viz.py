"""Спільна теплова візуалізація: inferno LUT + colorize().

Використовується record_session.py і трекерами керівника (SORT/_001/_002),
щоб не дублювати таблицю кольорів і функцію мапінгу.

Швидкий приклад:
    from thermal_viz import colorize
    bgr = colorize(frame2d, t_min=20.0, t_max=60.0, disp_w=640, disp_h=480)
"""
from __future__ import annotations

import numpy as np
import cv2

# Inferno keyframes у RGB (з matplotlib, без залежності від нього).
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


def inferno_lut(channel_order: str = "bgr") -> np.ndarray:
    """256-входова look-up table, dtype=uint8, shape=(256, 3).

    channel_order:
        "bgr" — для OpenCV (cv2.imencode/imshow/VideoWriter); за замовч.
        "rgb" — якщо плануєш конвертувати власноруч.
    """
    t   = np.linspace(0, 1, 256)
    tk  = np.linspace(0, 1, len(_INFERNO_KEY))
    cols = [np.interp(t, tk, _INFERNO_KEY[:, i]) for i in range(3)]
    rgb = np.stack(cols, axis=1)                    # (256, 3) RGB
    lut = rgb[:, ::-1] if channel_order == "bgr" else rgb
    return (lut * 255).clip(0, 255).astype(np.uint8)


# Кешований LUT для звичайного OpenCV-шляху (BGR).
LUT_BGR = inferno_lut("bgr")


def colorize(frame2d: np.ndarray,
             t_min: float,
             t_max: float,
             disp_w: int,
             disp_h: int,
             interp: int = cv2.INTER_NEAREST) -> np.ndarray:
    """Перетворити float-кадр (H,W) на uint8 BGR (disp_h, disp_w).

    Шкала фіксована: значення t_min → колір 0 (майже чорний),
    t_max → колір 255 (жовто-білий); все за межами зрізається (clip).

    interp:
        cv2.INTER_NEAREST — чіткі пікселі, без блюру (за замовч.).
        cv2.INTER_CUBIC   — згладжено, але «розмиває» межі гарячих об'єктів.
    """
    span = max(float(t_max) - float(t_min), 1e-3)
    idx = ((frame2d - t_min) / span * 255).clip(0, 255).astype(np.uint8)
    small = LUT_BGR[idx]                                  # (H, W, 3) BGR
    return cv2.resize(small, (disp_w, disp_h), interpolation=interp)
