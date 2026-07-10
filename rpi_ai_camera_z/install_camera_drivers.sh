#!/bin/bash
# Встановлення всіх необхідних залежностей на Raspberry Pi

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Визначити Python у .venv, якщо він є
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python3"
elif [ -x "$SCRIPT_DIR/.venv/bin/python3" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
else
    PYTHON_BIN="python3"
fi

PIP_BIN="$PYTHON_BIN -m pip"

echo "╔════════════════════════════════════════════╗"
echo "║   Installing Camera Drivers & Libraries   ║"
echo "╚════════════════════════════════════════════╝"
echo

echo "🧪 Python interpreter: $PYTHON_BIN"
echo

# Оновити репозиторії
echo "📦 Оновлення репозиторіїв..."
sudo apt-get update

# Встановити основні пакети
echo "📦 Встановлення libcamera-tools..."
sudo apt-get install -y libcamera-tools libcamera-dev

# Встановити системні заголовки, необхідні для збірки Python-модулів
# Це вирішує помилку про відсутність libcap development headers
echo "📦 Встановлення розробницьких пакетів..."
sudo apt-get install -y build-essential libcap-dev pkg-config

# Встановити системні пакети для libcamera
# (це потрібно для нормальної роботи picamera2)
echo "📦 Встановлення Python libcamera/"
sudo apt-get install -y python3-libcamera python3-picamera2

# Встановити пакети у віртуальне середовище, якщо воно є
# Це найкраще працює для .venv/ на Raspberry Pi
if [ -n "$VIRTUAL_ENV" ] || [ -x "$SCRIPT_DIR/.venv/bin/python3" ]; then
    echo "📦 Встановлення picamera2 у віртуальне середовище..."
    $PYTHON_BIN -m pip install --upgrade pip setuptools wheel
    $PYTHON_BIN -m pip install picamera2
fi

# Встановити додаткові інструменти
echo "📦 Встановлення v4l-utils..."
sudo apt-get install -y v4l-utils

# Додати користувача до групи video
echo "👤 Додаю користувача до групи 'video'..."
sudo usermod -a -G video $(whoami)

echo
echo "✅ Встановлення завершено!"
echo
echo "⚠️  ВАЖЛИВО: Теб потрібно перезавантажитися або перезаттачити сеанс:"
echo "   newgrp video"
echo "   або перезавантажити: sudo reboot"
echo
echo "📝 Після цього спробуй:"
echo "   python3 ai_camera_stream_libcamera.py"
echo "   або діагностику:"
echo "   bash diagnose_camera.sh"
