"""HUD-візуалізація: малювання елементів інтерфейсу на overlay."""

import numpy as np
import cv2

from .models import Detection
from . import config


def create_overlay(width: int, height: int) -> np.ndarray:
    """Порожній чорний кадр (чорне = прозоре в overlay режимі)."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def draw_crosshair(frame: np.ndarray, cx: int, cy: int) -> None:
    cv2.line(frame, (cx - config.CROSSHAIR_SIZE, cy), (cx + config.CROSSHAIR_SIZE, cy),
             config.CROSSHAIR_COLOR, config.CROSSHAIR_THICKNESS)
    cv2.line(frame, (cx, cy - config.CROSSHAIR_SIZE), (cx, cy + config.CROSSHAIR_SIZE),
             config.CROSSHAIR_COLOR, config.CROSSHAIR_THICKNESS)


def draw_mode(frame: np.ndarray, mode: str) -> None:
    mode_colors = {"MANUAL": (0, 255, 0), "LOCK": (0, 255, 255), "AUTO": (0, 0, 255), "BRAKE": (255, 0, 255)}
    color = mode_colors.get(mode, (255, 255, 255))
    cv2.putText(frame, mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)


def draw_detections(frame: np.ndarray, detections: list[Detection],
                    locked_id: int | None = None) -> None:
    """Малює рамки всіх детекцій. Залоковану підсвічує іншим кольором."""
    for det in detections:
        is_locked = locked_id is not None and det.track_id == locked_id
        col = config.BBOX_COLOR_LOCKED if is_locked else config.BBOX_COLOR
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), col, config.BBOX_THICKNESS)
        cv2.putText(frame, f"car {det.confidence:.0%} ID:{det.track_id}",
                    (det.x1, det.y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)


def draw_lock_info(frame: np.ndarray, target: Detection, cx_screen: int, cy_screen: int,
                   locked_id: int) -> None:
    """LOCK: лінія до цілі + підпис."""
    cx_t, cy_t = target.center
    cv2.line(frame, (cx_screen, cy_screen), (cx_t, cy_t), (0, 255, 255), config.LINE_THICKNESS)
    cv2.putText(frame, f"LOCKED ID:{locked_id} [ПРОБІЛ = АТАКА]",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.circle(frame, (cx_t, cy_t), 5, (0, 255, 255), -1)


def draw_auto_info(frame: np.ndarray, target: Detection, cx_screen: int, cy_screen: int,
                   locked_id: int, attack_phase: str, screen_width: int) -> None:
    """AUTO: лінія + фаза + прогрес-бар наближення."""
    cx_t, cy_t = target.center
    bbox_ratio = target.width / max(screen_width, 1)

    cv2.line(frame, (cx_screen, cy_screen), (cx_t, cy_t), config.LINE_COLOR, config.LINE_THICKNESS)

    phase_colors = {
        "SEARCH": (128, 128, 128),
        "APPROACH": (0, 255, 255),
        "ATTACK": (0, 165, 255),
        "TERMINAL": (0, 0, 255),
    }
    ph_color = phase_colors.get(attack_phase, (255, 255, 255))

    cv2.putText(frame, f"LOCKED ID:{locked_id}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    cv2.putText(frame, f"PHASE: {attack_phase}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ph_color, 2)
    cv2.putText(frame, f"BBOX: {bbox_ratio:.0%}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ph_color, 2)

    # Прогрес-бар наближення
    bar_y, bar_w, bar_h = 140, 200, 16
    fill = min(1.0, bbox_ratio / config.PHASE_TERMINAL_RATIO)
    cv2.rectangle(frame, (10, bar_y), (10 + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    cv2.rectangle(frame, (10, bar_y), (10 + int(bar_w * fill), bar_y + bar_h), ph_color, -1)
    cv2.putText(frame, "IMPACT", (10 + bar_w + 5, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ph_color, 1)

    cv2.circle(frame, (cx_t, cy_t), 5, ph_color, -1)


def draw_brake_info(frame: np.ndarray, flow_x: float, flow_y: float,
                    flow_div: float) -> None:
    """BRAKE: показує 3 компоненти optical flow і стан гальмування."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    color = (255, 0, 255)  # magenta

    cv2.putText(frame, "BRAKE [B = exit]",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Roll (X): {flow_x:+.3f}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(frame, f"Thr  (Y): {flow_y:+.3f}",
                (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(frame, f"Pitch(Z): {flow_div:+.3f}",
                (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Вектор бокового + вертикального зміщення
    arrow_scale = 300
    end_x = int(cx + flow_x * arrow_scale)
    end_y = int(cy + flow_y * arrow_scale)
    cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 3, tipLength=0.3)

    # Індикатор forward/backward (кільце: розширюється = вперед, стискається = назад)
    ring_radius = int(40 + flow_div * 80)
    ring_radius = max(10, min(120, ring_radius))
    ring_color = (0, 200, 255) if flow_div > 0.05 else (255, 200, 0) if flow_div < -0.05 else color
    cv2.circle(frame, (cx, cy), ring_radius, ring_color, 2)
