from multiprocessing import freeze_support
from ultralytics import YOLO
import os
import cv2
import yaml
from pathlib import Path

if __name__ == '__main__':
    freeze_support()
    
    MODEL_PATH = "models/21/35.pt"                  
    DATA_YAML = "datasets/test_1_3x/data.yaml"       
    OUTPUT_DIR = "ev_40" 
    RUN_VISUALIZATION = True
    OUTPUT_PATH = Path("runs/detect") / (OUTPUT_DIR + "_visualized")
    CONF_THRESH = 0.25 

    # Кольори (BGR): Зелений для Ground Truth, Червоний для Predictions
    GT_COLOR = (0, 255, 0)
    PRED_COLOR = (0, 0, 255)


    model = YOLO(MODEL_PATH)

    print("=== 1. ОБЧИСЛЕННЯ МЕТРИК ТОЧНОСТІ ===")
    metrics = model.val(data=DATA_YAML, split="test", name=OUTPUT_DIR)
    # metrics = model.val(data="datasets/own/data.yaml", split="test",
    #     plots=True,          # Зберігає Confusion Matrix, криві PR/F1 та приклад батчів
    #     save_json=True,      # Зберігає результати у форматі JSON (стандарт COCO)
    #     project="runs/detect",
    #     name="test_results"  # Фіксована назва папки з результатами)
    # )
    
    
    if not RUN_VISUALIZATION:
        exit()

    # 3. Зчитування конфігурації датасету
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Читаємо назви класів
    names = data_cfg.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    # Знаходимо папку з тестовими зображеннями
    data_yaml_path = Path(DATA_YAML).resolve()
    test_path = Path(data_cfg.get("test", "test/images"))
    images_dir = test_path if test_path.is_absolute() else data_yaml_path.parent / test_path
    if not images_dir.is_dir() and not test_path.is_absolute():
        images_dir = data_yaml_path.parent / Path(*test_path.parts[1:])
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Test images directory not found: {images_dir}")

    # Отримуємо список усіх тестових фото
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in valid_exts]

    print(f"=== 2. ОБРОБКА ТА ВІЗУАЛІЗАЦІЯ ({len(image_files)} ЗОБРАЖЕНЬ) ===")

    # 4. Прохід по кожному зображенню
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_h, img_w = img.shape[:2]

        # --- А) Малюємо Ground Truth (Зелений) ---
        # Знаходимо відповідний .txt файл міток (images/test -> labels/test)
        label_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
        
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:5])
                        
                        # Денормалізація координат YOLO -> пікселі
                        x1 = int((x_c - w / 2) * img_w)
                        y1 = int((y_c - h / 2) * img_h)
                        x2 = int((x_c + w / 2) * img_w)
                        y2 = int((y_c + h / 2) * img_h)
                        
                        cls_name = names.get(cls_id, str(cls_id))
                        
                        # Рамка та підпис для GT (зверху рамки)
                        cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, 2)
                        cv2.putText(
                            img, f"GT: {cls_name}", (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 2
                        )

        # --- Б) Малюємо Predictions (Червоний) ---
        results = model.predict(source=str(img_path), conf=CONF_THRESH, verbose=False)[0]
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            cls_name = names.get(cls_id, str(cls_id))
            
            # Рамка та підпис для Pred (знизу рамки, щоб текстові мітки не накладалися)
            cv2.rectangle(img, (x1, y1), (x2, y2), PRED_COLOR, 2)
            cv2.putText(
                img, f"Pred: {cls_name} {conf:.2f}", (x1, min(y2 + 18, img_h - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRED_COLOR, 2
            )

        # Збереження результату
        cv2.imwrite(str(OUTPUT_PATH / img_path.name), img)

    print(f"\nГотово! Усі візуалізовані кадри збережено у: {OUTPUT_PATH.resolve()}")