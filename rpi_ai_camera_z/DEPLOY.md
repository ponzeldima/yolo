# Розгортування AI Camera Stream на Raspberry Pi через SSH

## Швидкий старт (5 хвилин)

### 1. Скопіюй папку на Pi
```bash
scp -r rpi_ai_camera_z pi@raspberrypi.local:/home/pi/ai_camera_stream
```

### 2. Запусти setup
```bash
ssh pi@raspberrypi.local
cd ~/ai_camera_stream
chmod +x setup.sh run.sh
./setup.sh
```

### 3. Запусти стрімер
```bash
./run.sh
```

Твій браузер (з ПК/ноутбука):
```
http://raspberrypi.local:8080
```

---

## Деталі по компонентам

### `ai_camera_stream.py` (основний скрипт)
- Читає з веб-камери через OpenCV
- YOLOv8 детекція в реальному часі (за замовчуванням)
- MJPEG HTTP стрімер на `localhost:8080`
- Контроль через браузер (детекція on/off, трекінг, quit)
- FPS лічильник і статус на екрані

**Параметри:**
```bash
python3 ai_camera_stream.py --help

--model MODEL         Шлях до YOLO моделі (default: yolov8n.pt)
--device DEVICE       ID камери (default: 0)
--port PORT           HTTP порт (default: 8080)
--conf CONF           Поріг конфіденції (default: 0.5)
--no-detection        Вимкнути детекцію
```

### `run.sh` (запускач)
Просто запускає основний скрипт з активованим `.venv`. 

**Зроби виконуваним на Pi:**
```bash
chmod +x run.sh
```

### `setup.sh` (ініціалізація)
- Створює `.venv`
- Встановлює залежності з `requirements.txt`
- Завантажує YOLOv8 модель
- Перевіряє камеру

**Одноразово при першому клонуванні:**
```bash
chmod +x setup.sh
./setup.sh
```

### `ai-camera.service` (systemd сервіс)
Запускати стрімер як фоновий сервіс (автозапуск при реботі).

**Інсталяція:**
```bash
sudo cp ai-camera.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-camera
sudo systemctl start ai-camera

# Перевірка
sudo systemctl status ai-camera

# Логи
sudo journalctl -u ai-camera -f
```

---

## Камера на Pi

Камера повинна бути підключена до `cam/disp1` через CnDC Standard-mini.

**Перевірка доступу:**
```bash
ls -la /dev/video*
```

Повинна бути `/dev/video0` або `/dev/video1` (залежить від того, скільки камер).

**Якщо не знаходиш камеру:**
```bash
# Спробуй яка-небудь інша:
python3 ai_camera_stream.py --device 1

# Встанови інструменти
sudo apt-get install v4l-utils
v4l2-ctl --list-devices
```

---

## Налаштування детекції

### Змінити модель
```bash
# Nano (швидко, мало памяті, менш точна)
./run.sh --model yolov8n.pt

# Small (баланс)
./run.sh --model yolov8s.pt

# Medium (точніша, потребує більше памяті)
./run.sh --model yolov8m.pt
```

### Змінити поріг конфіденції
```bash
# 0.5 - за замовчуванням (більше детекцій)
./run.sh --conf 0.5

# 0.7 - більш вибіркова
./run.sh --conf 0.7
```

### Вимкнути детекцію
```bash
./run.sh --no-detection
```

---

## Веб-інтерфейс (http://raspberrypi.local:8080)

| Кнопка | Функція |
|--------|---------|
| 🎯 Детекція: ON | Toggle YOLOv8 детекції об'єктів |
| 📍 Трекінг: OFF | (зарезервовано для майбутнього трекінгу) |
| 📊 Скинути FPS | Обнули FPS лічильник |
| ❌ Quit | Завершити скрипт |

На екрані також показується:
- Поточний FPS
- Статус детекції
- Статус трекінгу
- Bounding boxes детектованих обєктів

---

## Розширення і експерименти

### Запис сесій з детекціями
Можна додати код до `ai_camera_stream.py` для запису кадрів і детекцій 
(як у `rpi_thermal_tracking_z/record_session.py`):

```python
# Додай до _process_frame():
frame_dict = {
    'frame': frame,
    'detections': results[0].boxes.cpu().numpy(),
    'timestamp': time.time()
}
# Запиши у HDF5 або CSV
```

### DeepSORT трекінг
Можна інтегрувати трекер з папки `rpi_thermal_tracking_z/trackers/`:

```python
from trackers.deepsort_cnn import DeepSORTCNNTracker
tracker = DeepSORTCNNTracker(...)
# У циклі:
tracked_objects = tracker.update(detections)
```

### REST API для запита детекцій
Додай endpoint для REST запитів:

```python
@_HTTPHandler.do_POST
def handle_detection_query(self):
    # Повернути JSON з останніми детекціями
    ...
```

---

## Проблеми і рішення

### Камера не відкривається
```bash
# 1. Перевір доступ
ls -la /dev/video*

# 2. Спробуй інший device
./run.sh --device 1

# 3. Перевір дозволи (можливо потрібен sudo)
sudo python3 ai_camera_stream.py
```

### Повільний стрім
```bash
# 1. Вимкни детекцію
./run.sh --no-detection

# 2. Використай nano модель
./run.sh --model yolov8n.pt

# 3. Збільш поріг конфіденції (менше обчислень)
./run.sh --conf 0.7

# 4. Перевір мережу (ping Raspberry Pi)
ping raspberrypi.local
```

### Помилка при завантаженні моделі
```bash
# Скопіюй yolov8n.pt з цієї машини
scp yolov8n.pt pi@raspberrypi.local:~/ai_camera_stream/
./run.sh --model ./yolov8n.pt
```

### Якщо запустив `setup.sh` і це не допомогло
```bash
# Запусти setpu ще раз з sudo (для доступу до камери)
sudo ./setup.sh

# Встанови додаткові пакети
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip libatlas-base-dev libjasper-dev \
    libtiff5 libjasper1 libharfbuzz0b libwebp6 libtiff5 libjasper1 libharfbuzz0b
```

---

## Структура папки

```
rpi_ai_camera_z/
├── ai_camera_stream.py      ← основний скрипт
├── setup.sh                 ← першопочатковий setup
├── run.sh                   ← запускач з .venv
├── requirements.txt         ← залежності (pip install)
├── ai-camera.service        ← systemd сервіс (опційно)
├── README.md                ← цей файл
└── DEPLOY.md                ← цей файл
```

---

**Готово! Приступай до запуску! 🚀**
