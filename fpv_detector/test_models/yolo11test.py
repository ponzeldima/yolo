from ultralytics import YOLO
import os

model = YOLO('/Users/dmytroponzel/Desktop/yolo/fpv_detector/test_models/weights/best_own.pt')
# model = YOLO('yolov8n.pt')  # Завантажуєте модель YOLOv8n
# Вказуєте шлях до завантаженого файлу data.yaml
metrics = model.val(data="/Users/dmytroponzel/Desktop/yolo/fpv_detector/datasets/test_own/data.yaml", split="test")