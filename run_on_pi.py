from ultralytics import YOLO
import time

# Завантажуємо модель (pt або ncnn)
model = YOLO('best_ncnn_model') 

# Запуск детекції
# save=True — зберігає результат у файл
# imgsz=320 — для швидкості
# conf=0.25 — поріг впевненості
results = model.predict(
    source='shahed_test_6.mp4', 
    save=True, 
    imgsz=640, 
    stream=True, 
    verbose=False
)

print("Processing started...")

for i, r in enumerate(results):
    # Виводимо FPS в консоль кожні 10 кадрів
    if i % 10 == 0:
        inference_time = r.speed['inference']
        fps = 1000 / inference_time
        print(f"Frame {i}: Inference {inference_time:.1f}ms ({fps:.2f} FPS)")

print("Processing finished!")