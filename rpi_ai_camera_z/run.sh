#!/bin/bash
# Простий скрипт для запуску на Raspberry Pi з вже ініціалізованим .venv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/.venv"

# Перевіри .venv
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Віртуальне середовище не знайдено!"
    echo "Створи його:"
    echo "  python3 -m venv $VENV_PATH"
    echo "  source $VENV_PATH/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Активуй .venv
source "$VENV_PATH/bin/activate"

# Запусти скрипт з аргументами
python3 "$SCRIPT_DIR/ai_camera_stream.py" "$@"
