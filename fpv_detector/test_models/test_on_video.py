import time

import cv2
from ultralytics import YOLO
import torch
import onnxruntime as ort

# Load CUDA 13.x / cuDNN DLLs from NVIDIA Python packages
# ort.preload_dlls()

model = YOLO("models/9/10.pt")

results = model.predict(
    source="test_models/videos/1.mp4",
    device=0,
    show=False,
    stream=True,
)

start_time = time.perf_counter()
frame_count = 0
for r in results:
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

elapsed = time.perf_counter() - start_time
avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
print(f"Average FPS: {avg_fps:.1f} over {frame_count} frames ({elapsed:.1f}s)")

# Завантажуємо попередньо навчену модель YOLOv8 (наприклад, nano-версію)
# model = YOLO("train_models/runs/detect/v8n_1/weights/best.pt")  # Вкажіть шлях до вашої моделі

# # Відкриваємо відеофайл (або вкажіть 0 для веб-камери)
# video_path = "test_models/videos/fpv_3.mov"
# cap = cv2.VideoCapture(video_path)

# while cap.isOpened():
#   success, frame = cap.read()
#   if not success:
#     break

#   # Запускаємо передбачення моделі на повному кадрі
#   results = model(frame)

#   # Візуалізуємо результати (накладаємо рамки) на кадр
#   annotated_frame = results[0].plot()

#   # Показуємо результат у вікні
#   cv2.imshow("YOLO Live", annotated_frame)

#   # Натисніть 'q', щоб вийти з циклу
#   if cv2.waitKey(1) & 0xFF == ord("q"):
#     break

# cap.release()
# cv2.destroyAllWindows()
