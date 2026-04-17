"""
Система візуального автонаведення дрона на ціль (автомобіль) в симуляторі Uncrashed.

Запуск (рекомендовано):
    python -m uncrashed_tracker

Або напряму:
    python uncrashed_tracker/drone_visual_aim.py

Залежності:
    pip install dxcam ultralytics opencv-python numpy pywin32 vgamepad keyboard pygame
    + Драйвер ViGEmBus: https://github.com/nefarius/ViGEmBus/releases

Запуск від адміністратора (для keyboard).
"""

import sys
import os

# Додаємо батьківську директорію в sys.path для коректного імпорту пакету
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uncrashed_tracker.__main__ import main

if __name__ == "__main__":
    main()
