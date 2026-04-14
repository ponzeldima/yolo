"""
Fine-tune YOLOv8n на датасеті uncrashed_cars (Roboflow).
Після тренування — експорт в TensorRT (.engine).

Запуск:
    python train_uncrashed.py
"""

from ultralytics import YOLO

# ── Налаштування ──
BASE_MODEL = "yolov8n.pt"                                    # pretrained база
DATA_YAML = "datasets/uncrashed_cars.v2i.yolov8/data.yaml"            # шлях до датасету
EPOCHS = 100                                                  # кількість епох
IMGSZ = 640                                                   # розмір інференсу (як у drone_visual_aim)
BATCH = 16                                                    # batch size (зменши до 8 якщо не вистачить VRAM)
PROJECT = "runs/detect"
NAME = "uncrashed_cars"

# ── Тренування ──
if __name__ == "__main__":
    model = YOLO(BASE_MODEL)
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device="cuda",
        project=PROJECT,
        name=NAME,
        patience=20,       # early stopping: зупинка якщо val loss не покращується 20 епох
        save=True,
        save_period=10,    # зберігати чекпоінт кожні 10 епох
        plots=True,        # графіки тренування
        workers=0,         # без multiprocessing (Windows spawn issue)
    )

    # ── Експорт найкращої моделі в TensorRT ──
    best_path = f"{PROJECT}/{NAME}/weights/best.pt"
    print(f"\n[INFO] Експорт {best_path} → TensorRT (.engine)")
    best = YOLO(best_path)
    best.export(format="engine", imgsz=IMGSZ, half=True)
    print("[INFO] Готово! Файл: runs/detect/uncrashed_cars/weights/best.engine")
    print("[INFO] В drone_visual_aim.py зміни:")
    print(f'  MODEL_PATH = "{PROJECT}/{NAME}/weights/best.engine"')
    print(f"  CAR_CLASS_ID = 0")
