# AI Camera Stream для Raspberry Pi

MJPEG стрімер з YOLOv8 детекцією для AI камери (aitrois).

## Встановлення на Raspberry Pi

```bash
# 1. Скопіюй папку на Pi
scp -r ai_camera_stream pi@raspberrypi.local:/home/pi/

# 2. Перейди в папку
ssh pi@raspberrypi.local
cd ~/ai_camera_stream

# 3. Встанови залежності (або використовуй існуючий .venv)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Якщо на Pi вже є проект yolo, скопіюй звідти yolov8n.pt:
cp ../yolo/yolov8n.pt .
```

## Запуск

```bash
# З активованим .venv
source .venv/bin/activate

# Базовий запуск (детекція ON за замовчуванням)
python3 ai_camera_stream.py

# Лише стрім без детекції
python3 ai_camera_stream.py --no-detection

# Інша YOLO модель (більша, точніша)
python3 ai_camera_stream.py --model yolov8m.pt

# Інші опції
python3 ai_camera_stream.py --help
```

## Доступ з браузера

Після запуску скрипта йди на:
```
http://raspberrypi.local:8080
```

або якщо не працює DNS:
```
http://<IP_МАЛИНКИ>:8080
```

## Веб-інтерфейс

- **🎯 Детекція: ON/OFF** - включи/вимкни YOLOv8 детекцію об'єктів
- **📍 Трекінг: OFF** - резервна кнопка для майбутнього трекінгу
- **📊 Скинути FPS** - обнули лічильник FPS
- **❌ Quit** - завершити скрипт

## Підключення камери

Камера підключена через:
- **CnDC Standard-mini** до **cam/disp1** на Raspberry Pi

## Форма запису кадрів

За потреби додай до скрипту запис кадрів і анотацій (як у thermal tracking):
- `.f32` - сирі кадри (float32)
- `.f64` - timestamps
- `.json` - metadata
- `.csv` - детекції в CSV

## Налаштування детекції

Змінити поріг конфіденції:
```bash
python3 ai_camera_stream.py --conf 0.6
```

Більший поріг = менше помилкових детекцій, але може пропустити обєкти.

## Моделі YOLO

- `yolov8n.pt` - nano (швидка, мало памяті) ✓ рекомендується для Pi
- `yolov8s.pt` - small (баланс)
- `yolov8m.pt` - medium (точніша, але повільніше)
- Завантажуються автоматично при першому запуску

## Проблеми

**Камера не відкривається:**
```bash
# Перевір доступ
ls -la /dev/video*

# Спробуй device 1 або 2
python3 ai_camera_stream.py --device 1
```

**Повільний стрім:**
- Зменш якість JPEG в коді (зараз 80)
- Використай меншу модель: `yolov8n.pt`
- Вимкни детекцію: `--no-detection`
- Проверь зв'язок мережі

**Помилка YOLO моделі:**
```bash
# Скопіюй yolov8n.pt від себе на Pi
scp yolov8n.pt pi@raspberrypi.local:~/ai_camera_stream/
python3 ai_camera_stream.py --model ./yolov8n.pt
```

## Розширення

Розширення функціоналу:
1. Запис сесій з детекціями (як thermal tracking)
2. DeepSORT трекінг об'єктів
3. REST API для запита детекцій
4. WebSocket для низколатентного стріму
5. Інтеграція з дроном RС контролем

---

**Автор:** Для проекту з дронами  
**Остання зміна:** 2024-2025
