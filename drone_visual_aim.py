"""
Система візуального автонаведення дрона (Етап 1: Комп'ютерний зір + обчислення похибки).

Захоплює екран macOS через mss, розпізнає машини (YOLO v8n),
обчислює відхилення центру цілі від центру екрану (Delta X / Delta Y)
та візуалізує результат через OpenCV.

Залежності:
    pip install mss ultralytics opencv-python numpy
"""

import time
import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

# ── Налаштування ────────────────────────────────────────────────────────────

# Область захоплення екрану (bounding box).
# None — весь головний монітор; або задай dict: {"top": 100, "left": 100, "width": 1280, "height": 720}
CAPTURE_REGION: dict | None = None

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

# ── Заглушка для наступного етапу (віртуальний джойстик) ─────────────────────


def send_commands_to_simulator(delta_x: float, delta_y: float) -> None:
    """TODO in next step — відправка команд на віртуальний геймпад / ПІД-регулятор."""
    pass


# ── Допоміжні функції ────────────────────────────────────────────────────────


def draw_crosshair(frame: np.ndarray, cx: int, cy: int) -> None:
    """Малює перехрестя (приціл) у точці (cx, cy)."""
    cv2.line(frame, (cx - CROSSHAIR_SIZE, cy), (cx + CROSSHAIR_SIZE, cy),
             CROSSHAIR_COLOR, CROSSHAIR_THICKNESS)
    cv2.line(frame, (cx, cy - CROSSHAIR_SIZE), (cx, cy + CROSSHAIR_SIZE),
             CROSSHAIR_COLOR, CROSSHAIR_THICKNESS)


def pick_best_car(results) -> tuple | None:
    """Повертає (x1, y1, x2, y2, conf) найвпевненішої детекції car або None."""
    best = None
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id == CAR_CLASS_ID and conf >= CONFIDENCE_THRESHOLD:
            if best is None or conf > best[4]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                best = (int(x1), int(y1), int(x2), int(y2), conf)
    return best


# ── Головний цикл ───────────────────────────────────────────────────────────


def main() -> None:
    model = YOLO(MODEL_PATH)

    with mss() as sct:
        # Визначаємо область захоплення
        monitor = sct.monitors[1]  # головний монітор
        if CAPTURE_REGION is not None:
            region = CAPTURE_REGION
        else:
            region = {
                "top": monitor["top"],
                "left": monitor["left"],
                "width": monitor["width"],
                "height": monitor["height"],
            }

        # Центр області захоплення — наш «приціл»
        cx_screen = region["width"] // 2
        cy_screen = region["height"] // 2

        print(f"[INFO] Область захоплення: {region['width']}x{region['height']} "
              f"(top={region['top']}, left={region['left']})")
        print(f"[INFO] Центр прицілу: ({cx_screen}, {cy_screen})")
        print("[INFO] Натисни 'q' у вікні OpenCV для виходу.\n")

        while True:
            t0 = time.perf_counter()

            # 1. Захоплення кадру
            screenshot = sct.grab(region)
            frame = np.array(screenshot, dtype=np.uint8)
            # mss повертає BGRA → конвертуємо в BGR для OpenCV / YOLO
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # 2. Детекція (YOLO) — device="mps" для GPU на Apple Silicon
            results = model.predict(frame, verbose=False, conf=CONFIDENCE_THRESHOLD,
                                    classes=[CAR_CLASS_ID], device="cuda")

            # 3. Вибір найкращої цілі
            car = pick_best_car(results)

            # 4. Малюємо перехрестя
            draw_crosshair(frame, cx_screen, cy_screen)

            if car is not None:
                x1, y1, x2, y2, conf = car
                # Центр bounding box цілі
                cx_target = (x1 + x2) // 2
                cy_target = (y1 + y2) // 2

                # Похибка (у пікселях)
                delta_x = cx_target - cx_screen
                delta_y = cy_target - cy_screen

                # Рамка навколо машини
                cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)

                # Лінія приціл → ціль
                cv2.line(frame, (cx_screen, cy_screen), (cx_target, cy_target),
                         LINE_COLOR, LINE_THICKNESS)

                # Підпис на рамці
                label = f"car {conf:.0%}  dx={delta_x} dy={delta_y}"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, BBOX_COLOR, 2)

                # Вивід у консоль
                fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
                print(f"Delta X: {delta_x:+5d}  |  Delta Y: {delta_y:+5d}  |  "
                      f"conf: {conf:.0%}  |  FPS: {fps:.1f}")

                # ── Місце для наступного етапу ──
                send_commands_to_simulator(delta_x, delta_y)

            else:
                fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
                print(f"[NO TARGET]  FPS: {fps:.1f}", end="\r")

            # 5. Відображення
            cv2.imshow("Drone Visual Aim", frame)

            # Вихід по 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("\n[INFO] Завершено.")


if __name__ == "__main__":
    main()
