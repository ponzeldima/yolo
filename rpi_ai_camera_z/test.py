import os
import time
from datetime import datetime
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder

# 1. ЗАЙТИ СЮДИ І ПОЧЕКАТИ 5 СЕКУНД ПРИ ЗАПУСКУ
print("Очікування 5 секунд перед початком роботи камери...")
time.sleep(5)

# 2. Ініціалізація камери
picam = Picamera2()

# 3. Базове налаштування відеопотоку
video_config = picam.create_video_configuration(
    main={"size": (1920, 1080), "format": "XBGR8888"}
)
picam.configure(video_config)

# 4. ДИНАМІЧНА НАЗВА ФАЙЛУ З ДАТОЮ ТА ЧАСОМ
# Формат: РРРРММДД_ГГХХСС (наприклад: 20260710_134502_video.h264)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Вказуємо точний абсолютний шлях до вашої робочої папки
save_dir = "/mnt/usb_video"
output_filename = os.path.join(save_dir, f"{current_time}_video.h264")

print("Запуск прев'ю камери...")
picam.start()

print(f"Початок запису у файл {output_filename}...")

# 5. Створюємо енкодер (бітрейт 10 Мбіт/с)
encoder = H264Encoder(10000000)

# 6. Запускаємо запис
picam.start_recording(encoder, output_filename)

try:
    # Записуємо відео протягом 10 секунд (змініть це число, якщо треба довше)
    time.sleep(60)
finally:
    print("Зупинка запису...")
    picam.stop_recording()
    picam.stop()
    print("Відео успішно збережено!")