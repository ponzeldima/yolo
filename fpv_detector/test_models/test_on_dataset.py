from multiprocessing import freeze_support
from ultralytics import YOLO
import os

if __name__ == '__main__':
    freeze_support()

    model = YOLO('runs/detect/v8m_640_dataset_3/weights/best.pt')
    # model = YOLO('yolov8n.pt')  # Завантажуєте модель YOLOv8n
    # Вказуєте шлях до завантаженого файлу data.yaml
    metrics = model.val(data="datasets/dataset_3/data.yaml", split="val")
    # metrics = model.val(data="datasets/own/data.yaml", split="test",
    #     plots=True,          # Зберігає Confusion Matrix, криві PR/F1 та приклад батчів
    #     save_json=True,      # Зберігає результати у форматі JSON (стандарт COCO)
    #     project="runs/detect",
    #     name="test_results"  # Фіксована назва папки з результатами)
    # )