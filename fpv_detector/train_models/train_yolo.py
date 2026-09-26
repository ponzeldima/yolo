import os
from ultralytics import YOLO
import torch
print(torch.__version__)

FOLDER_NAME = "v8n_960_dataset_3"

def main():
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # Шлях до конфігурації вашого датасету (перевірте, щоб він був правильним)
    # mac os
    # dataset_yaml = "/Users/dmytroponzel/Desktop/yolo/fpv_detector/datasets/own/data.yaml"
    dataset_yaml = "datasets/dataset_3/data.yaml"
    
    # Шлях до останніх збережених ваг (назва папки має збігатися з параметром name у train)
    # mac os
    # last_weights_path = "runs/detect/drone_model/weights/last.pt"
    last_weights_path = f"runs/detect/{FOLDER_NAME}/weights/last.pt"

    # Перевіряємо, чи існує файл від попереднього (перерваного) тренування
    if os.path.exists(last_weights_path):
        print(f"Знайдено перерване тренування! Відновлюємо з {last_weights_path}...")
        # Завантажуємо останній збережений стан
        model = YOLO(last_weights_path)
        
        # Запускаємо train з параметром resume=True. 
        # Модель сама згадає всі налаштування, кількість епох і датасет.
        model.train(resume=True)
        
    else:
        print("Починаємо нове тренування з нуля...")
        # Завантажуємо чисту модель
        model = YOLO("yolov8n.pt")  # Використовуємо базову модель YOLOv8m
        
        model.train(
            data=dataset_yaml,
            epochs=100,
            imgsz=960,
            batch=32,
            device="cuda",  # Використовуємо GPU (якщо доступний)
            cache=True,        # Використовуємо кешування для прискорення тренування
            name=FOLDER_NAME, # Це ім'я папки, куди будуть зберігатися результати (і файл last.pt)
            save=True,
            plots=True,        # графіки тренування
            workers=0,       # без multiprocessing (Windows spawn issue)
            degrees=15.0,     # Поворот зображень до 15 градусів
            fliplr=0.5,       # Горизонтальний розворот (50% шанс)
            mosaic=1.0,       # Мозаїчна аугментація
        )

    print("Тренування повністю завершено!")

if __name__ == "__main__":
    main()