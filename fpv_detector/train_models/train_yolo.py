import os
from ultralytics import YOLO

def main():
    # Шлях до конфігурації вашого датасету (перевірте, щоб він був правильним)
    dataset_yaml = "/Users/dmytroponzel/Desktop/yolo/fpv_detector/datasets/fpv1/data.yaml"
    
    # Шлях до останніх збережених ваг (назва папки має збігатися з параметром name у train)
    last_weights_path = "runs/detect/drone_model/weights/last.pt"

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
        model = YOLO("yolov8s.pt") 
        
        model.train(
            data=dataset_yaml,
            epochs=50,
            imgsz=640,
            batch=8,
            name="drone_model" # Це ім'я папки, куди будуть зберігатися результати (і файл last.pt)
        )

    print("Тренування повністю завершено!")

if __name__ == "__main__":
    main()