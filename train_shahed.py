import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


from ultralytics import YOLO
import torch
print(torch.__version__)
# import tensorflow as tf

# --- Функція, що спрацьовує в кінці кожної епохи ---
def print_epoch_metrics(trainer):
    # Отримуємо словник з метриками
    metrics = trainer.metrics
    
    # Витягуємо основні показники (ключі можуть трохи відрізнятися в нових версіях, але зазвичай такі)
    # (B) означає Box (для детекції об'єктів)
    map50 = metrics.get("metrics/mAP50(B)", 0)
    precision = metrics.get("metrics/precision(B)", 0)
    recall = metrics.get("metrics/recall(B)", 0)
    
    current_epoch = trainer.epoch + 1
    total_epochs = trainer.epochs

    # Формуємо власний рядок виводу (кольоровий, якщо термінал підтримує)
    print(f"\n📊 [Епоха {current_epoch}/{total_epochs}] "
          f"Точність (P): {precision:.1%} | "
          f"Повнота (R): {recall:.1%} | "
          f"mAP@50: {map50:.1%}")

if __name__ == '__main__':
    
    # Перевірка GPU (просто для інформації)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    # print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))

    # 1. Завантажуємо модель
    model = YOLO('yolov8n.pt')  # Скачає ваги автоматично при першому запуску
    
    # Додаємо наш callback до моделі
    model.add_callback("on_fit_epoch_end", print_epoch_metrics)
    
    # 2. Вказуємо шлях до ВАШОГО локального data.yaml
    # Важливо: використовуйте повний шлях або переконайтеся, що файл поруч
    data_path = 'C:\\Users\\ponze\\Desktop\ML\\yolo\\dataset_test_244\\data.yaml' 

    # 3. Запускаємо навчання
    results = model.train(
        data=data_path,
        epochs=100,             # Для тесту почніть з 50
        imgsz=640,
        batch=16,              # Якщо мало відеопам'яті - ставте 8 або 4
        device=0,              # 0 для GPU, 'cpu' для процесора
        workers=0,             # Для локального запуску на Windows краще 0 (стабільніше)
        project='test_244_local',
        name='run1',
        plots=True
    )

    print("Навчання завершено!")
    
    # Запускаємо валідацію
    metrics = model.val()

    # --- Витягуємо конкретні цифри ---

    # mAP50 (Mean Average Precision при IoU 0.5) 
    # Це головний показник "крутості" моделі. 
    # Для FPV нам треба хоча б 0.6-0.7, ідеально > 0.8.
    print(f"mAP@50: {metrics.box.map50:.3f}")

    # mAP50-95 (Середнє за всіма порогами)
    # Показує, наскільки точно рамка облягає дрон.
    print(f"mAP@50-95: {metrics.box.map:.3f}")

    # Precision (Точність): Скільки знайдених об'єктів дійсно є Шахедами?
    # (Важливо, щоб не атакувати своїх або птахів)
    print(f"Precision: {metrics.box.mp:.3f}") # Mean Precision

    # Recall (Повнота): Скільки реальних Шахедів ми знайшли?
    # (Важливо, щоб не пропустити ціль)
    print(f"Recall: {metrics.box.mr:.3f}")    # Mean Recall