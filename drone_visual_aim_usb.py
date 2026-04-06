"""
Система візуального автонаведення дрона — захоплення відео з USB-камери (UVC).

Skyzone Cobra X підключається через USB і працює як UVC-камера.
Скрипт розпізнає машини (YOLOv8n), обчислює відхилення центру цілі
від центру кадру (error_x / error_y) та візуалізує результат.

Залежності:
    pip install ultralytics opencv-python numpy

Використання:
    python drone_visual_aim_usb.py
    Натисни 'q' у вікні OpenCV для виходу.
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO

# ── Налаштування ────────────────────────────────────────────────────────────

# Індекс USB-камери (Skyzone Cobra X). Спробуй 0, 1 або 2.
CAMERA_INDEX = 0

# Шлях до моделі
MODEL_PATH = "yolov8n.pt"

# COCO-індекс класу «car» = 2
CAR_CLASS_ID = 2

# Поріг впевненості детекції
CONFIDENCE_THRESHOLD = 0.35

# Колір перехрестя (BGR), товщина
CROSSHAIR_COLOR = (0, 255, 0)
CROSSHAIR_SIZE = 20
CROSSHAIR_THICKNESS = 2

# Колір рамки цілі (BGR)
BBOX_COLOR = (0, 0, 255)
BBOX_THICKNESS = 2

# Колір лінії «центр екрану → центр цілі»
LINE_COLOR = (255, 255, 0)
LINE_THICKNESS = 2


# ── Заглушка для керування апаратурою ────────────────────────────────────────


def send_to_control(error_x: float, error_y: float) -> None:
    """
    Заглушка — сюди пізніше вставиш логіку керування апаратурою
    (віртуальний геймпад, serial-команди, PID-регулятор тощо).

    Args:
        error_x: відхилення цілі від центру кадру по X (пікселі, + вправо).
        error_y: відхилення цілі від центру кадру по Y (пікселі, + вниз).
    """
    pass


# ── Допоміжні функції ────────────────────────────────────────────────────────


def draw_crosshair(frame: np.ndarray, cx: int, cy: int) -> None:
    """Малює перехрестя (приціл) у точці (cx, cy)."""
    cv2.line(frame, (cx - CROSSHAIR_SIZE, cy), (cx + CROSSHAIR_SIZE, cy),
             CROSSHAIR_COLOR, CROSSHAIR_THICKNESS)
    cv2.line(frame, (cx, cy - CROSSHAIR_SIZE), (cx, cy + CROSSHAIR_SIZE),
             CROSSHAIR_COLOR, CROSSHAIR_THICKNESS)


def pick_nearest_car(results, cx_screen: int, cy_screen: int) -> tuple | None:
    """
    Повертає (x1, y1, x2, y2, conf) найближчої до центру кадру машини,
    або None якщо жодної не знайдено.
    """
    best = None
    best_dist = float("inf")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == CAR_CLASS_ID and conf >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx_target = (x1 + x2) / 2
            cy_target = (y1 + y2) / 2
            dist = (cx_target - cx_screen) ** 2 + (cy_target - cy_screen) ** 2
            if dist < best_dist:
                best_dist = dist
                best = (int(x1), int(y1), int(x2), int(y2), conf)
    return best


# ── Головний цикл ───────────────────────────────────────────────────────────


def main() -> None:
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Не вдалося відкрити камеру з індексом {CAMERA_INDEX}.")
        print("        Спробуй змінити CAMERA_INDEX на 0, 1 або 2.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx_screen = frame_w // 2
    cy_screen = frame_h // 2

    print(f"[INFO] Камера #{CAMERA_INDEX}: {frame_w}x{frame_h}")
    print(f"[INFO] Центр прицілу: ({cx_screen}, {cy_screen})")
    print("[INFO] Натисни 'q' у вікні OpenCV для виходу.\n")

    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("[WARN] Не вдалося прочитати кадр, пропускаю...")
            continue

        # Детекція — device="mps" для Apple Silicon GPU
        results = model.predict(
            frame, verbose=False,
            conf=CONFIDENCE_THRESHOLD,
            classes=[CAR_CLASS_ID],
            device="mps",
        )

        # Вибір найближчої до центру цілі
        car = pick_nearest_car(results, cx_screen, cy_screen)

        # Перехрестя у центрі кадру
        draw_crosshair(frame, cx_screen, cy_screen)

        if car is not None:
            x1, y1, x2, y2, conf = car
            cx_target = (x1 + x2) // 2
            cy_target = (y1 + y2) // 2

            error_x = cx_target - cx_screen
            error_y = cy_target - cy_screen

            # Рамка навколо машини
            cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)

            # Лінія приціл → ціль
            cv2.line(frame, (cx_screen, cy_screen), (cx_target, cy_target),
                     LINE_COLOR, LINE_THICKNESS)

            # Підпис
            label = f"car {conf:.0%}  ex={error_x} ey={error_y}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, BBOX_COLOR, 2)

            fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
            print(f"error_x: {error_x:+5d}  |  error_y: {error_y:+5d}  |  "
                  f"conf: {conf:.0%}  |  FPS: {fps:.1f}")

            # Передача похибки у функцію керування
            send_to_control(error_x, error_y)

        else:
            fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
            print(f"[NO TARGET]  FPS: {fps:.1f}", end="\r")

        cv2.imshow("Drone Visual Aim (USB)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Завершено.")


if __name__ == "__main__":
    main()
