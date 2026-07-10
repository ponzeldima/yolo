#!/bin/bash
# Setup скрипт для Raspberry Pi - виконувати один раз при першому клоніруванні

set -e

echo "╔════════════════════════════════════════════╗"
echo "║   AI Camera Stream Setup for Raspberry Pi  ║"
echo "╚════════════════════════════════════════════╝"
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/.venv"

# Крок 1: Перевір Python
echo "1️⃣ Перевірка Python версії..."
python3 --version
echo

# Крок 2: Створи .venv
if [ ! -d "$VENV_PATH" ]; then
    echo "2️⃣ Створюю віртуальне середовище..."
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip setuptools wheel
else
    echo "2️⃣ Віртуальне середовище вже існує"
    source "$VENV_PATH/bin/activate"
fi
echo

# Крок 3: Встанови залежності
echo "3️⃣ Встановлюю залежності з requirements.txt..."
pip install -r "$SCRIPT_DIR/requirements.txt"
echo

# Крок 4: Перевір kamery
echo "4️⃣ Перевіряю наявність камер..."
if command -v v4l2-ctl &> /dev/null; then
    echo "📹 Знайдені пристрої:"
    v4l2-ctl --list-devices || true
else
    echo "⚠️  v4l2-ctl не встановлений. Встанови: sudo apt-get install v4l-utils"
    ls -la /dev/video* 2>/dev/null || echo "   (немає /dev/video*)"
fi
echo

# Крок 5: Завантаж модель YOLO
echo "5️⃣ Завантажую YOLOv8 модель (може зайняти час)..."
python3 << 'EOF'
try:
    from ultralytics import YOLO
    print("   Завантажую yolov8n...")
    model = YOLO('yolov8n.pt')
    print("   ✓ Модель завантажена і збережена")
except Exception as e:
    print(f"   ⚠️ Помилка при завантаженні: {e}")
    print("   (можна спробувати пізніше під час першого запуску)")
EOF
echo

# Крок 6: Тестовий запуск
echo "6️⃣ Готівність до запуску:"
echo "   Для запуску скрипта використовуй:"
echo ""
echo "   ./run.sh                    # базовий запуск"
echo "   ./run.sh --no-detection     # без детекції"
echo "   ./run.sh --device 1         # інша камера"
echo ""
echo "   Або вручну:"
echo "   source .venv/bin/activate"
echo "   python3 ai_camera_stream.py"
echo ""

echo "✅ Setup завершено! Приступай до ./run.sh"
