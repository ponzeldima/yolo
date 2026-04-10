"""
Тест віртуального Xbox-контролера для симулятора Uncrashed.

Створює віртуальний геймпад (ViGEmBus). Натискання ПРОБІЛ вмикає/вимикає
газ на 30%, щоб перевірити чи симулятор приймає команди.

Маппінг стіків (Mode 2 — стандарт дронів):
    Лівий стік Y  → Throttle  (газ)
    Лівий стік X  → Yaw       (поворот)
    Правий стік X → Roll      (нахил вліво/вправо)
    Правий стік Y → Pitch     (нахил вперед/назад)

Вимоги:
    1. Встановити драйвер ViGEmBus:
       https://github.com/nefarius/ViGEmBus/releases
    2. pip install vgamepad keyboard

Використання:
    python test_vgamepad.py          (запускати від адміна для keyboard)
    ПРОБІЛ — увімкнути/вимкнути газ 30%
    Ctrl+C  — вихід
"""

import time
import keyboard
import vgamepad as vg

THROTTLE_PCT = 0.30  # 30% газу (0.0 .. 1.0)
# Маппінг: 0% газу = -1.0 (стік вниз), 100% газу = +1.0 (стік вгору)
# 30% газу = -1.0 + 0.30 * 2.0 = -0.4
THROTTLE_STICK = -1.0 + THROTTLE_PCT * 2.0
THROTTLE_OFF  = -1.0  # стік повністю вниз = 0% газу


def main() -> None:
    print("[INFO] Створюю віртуальний Xbox 360 контролер...")
    gamepad = vg.VX360Gamepad()
    gamepad.update()
    time.sleep(1)

    throttle_on = False

    def toggle_throttle(_event=None):
        nonlocal throttle_on
        throttle_on = not throttle_on

    keyboard.on_press_key("space", toggle_throttle)

    print("[INFO] Контролер створений. Переключись на вікно Uncrashed!")
    print("[INFO] ПРОБІЛ — увімкнути/вимкнути газ 30%")
    print("[INFO] Ctrl+C — вихід\n")

    try:
        while True:
            throttle = THROTTLE_STICK if throttle_on else THROTTLE_OFF

            gamepad.left_joystick_float(x_value_float=0.0, y_value_float=throttle)
            gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
            gamepad.update()

            pct = (throttle + 1.0) / 2.0 * 100  # -1..+1 → 0..100%
            state = "ON " if throttle_on else "OFF"
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"  [{state}] Throttle: {pct:5.1f}%  |{bar}|", end="\r")

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\n[INFO] Зупинка — відпускаю всі стіки...")
        gamepad.left_joystick_float(x_value_float=0.0, y_value_float=-1.0)
        gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
        gamepad.update()
        keyboard.unhook_all()
        print("[DONE]")


if __name__ == "__main__":
    main()
