"""
Тест віртуального Xbox-контролера для Uncrashed (Windows).

Створює віртуальний Xbox 360 геймпад через vgamepad (ViGEmBus).
Uncrashed бачить його як справжній контролер.

Режими стіків (Mode 2 — найпоширеніший):
    Лівий стік:  Y = throttle (газ),  X = yaw (поворот)
    Правий стік: Y = pitch (нахил),   X = roll (крен)

Залежності:
    1. Встанови ViGEmBus драйвер: https://github.com/nefarius/ViGEmBus/releases
    2. pip install vgamepad keyboard

Використання:
    python test_drone_control.py

    Натисни:
      [1] — Тест осей (плавно рухає кожен стік)
      [2] — Злетіти → повисіти → сісти
      [3] — Квадрат у повітрі
      [q] — Вийти
"""

import time
import sys

try:
    import vgamepad as vg
except ImportError:
    print("ПОМИЛКА: vgamepad не встановлено.")
    print("  1. Встанови ViGEmBus: https://github.com/nefarius/ViGEmBus/releases")
    print("  2. pip install vgamepad")
    sys.exit(1)

try:
    import keyboard
except ImportError:
    print("ПОМИЛКА: keyboard не встановлено.")
    print("  pip install keyboard")
    sys.exit(1)


# ── Налаштування ─────────────────────────────────────────────────────────────

# Значення стіків від -1.0 до 1.0
# Throttle: 0.0 = середина (hover у Uncrashed), 1.0 = повний газ, -1.0 = мін. газ

THROTTLE_UP = 0.6       # газ для підйому (не повний, щоб не вилетіти)
THROTTLE_HOVER = 0.0    # утримання висоти (середина стіка)
THROTTLE_DOWN = -0.5    # газ для посадки
THROTTLE_IDLE = -1.0    # мотори на мінімум

PITCH_FORWARD = -0.4    # нахил вперед
PITCH_BACK = 0.4        # нахил назад
YAW_LEFT = -0.5         # поворот вліво
YAW_RIGHT = 0.5         # поворот вправо
ROLL_LEFT = -0.4        # крен вліво
ROLL_RIGHT = 0.4        # крен вправо


# ── Клас-обгортка для зручності ──────────────────────────────────────────────


class DroneController:
    """Обгортка над vgamepad для керування дроном (Mode 2)."""

    def __init__(self):
        self.pad = vg.VX360Gamepad()
        self.reset()
        print("[OK] Віртуальний Xbox 360 контролер створено.")
        print("     Перевір у Windows: Settings → Bluetooth & devices → 'Xbox 360 Controller'")

    def set_sticks(self, throttle: float = 0.0, yaw: float = 0.0,
                   pitch: float = 0.0, roll: float = 0.0):
        """
        Встановити значення всіх осей одночасно.

        Args:
            throttle: -1.0 (мін) .. 1.0 (макс газ)     — лівий стік Y
            yaw:      -1.0 (ліво) .. 1.0 (право)       — лівий стік X
            pitch:    -1.0 (вперед) .. 1.0 (назад)      — правий стік Y
            roll:     -1.0 (ліво) .. 1.0 (право)        — правий стік X
        """
        self.pad.left_joystick_float(x_value_float=yaw, y_value_float=throttle)
        self.pad.right_joystick_float(x_value_float=roll, y_value_float=pitch)
        self.pad.update()

    def reset(self):
        """Скинути всі осі в нуль."""
        self.set_sticks(0.0, 0.0, 0.0, 0.0)

    def hold(self, duration: float, throttle=0.0, yaw=0.0, pitch=0.0, roll=0.0,
             label: str = ""):
        """Утримувати задані значення осей протягом duration секунд."""
        if label:
            print(f"  {label} ({duration:.1f} с) ...", flush=True)
        self.set_sticks(throttle, yaw, pitch, roll)
        time.sleep(duration)

    def smooth_transition(self, duration: float, steps: int = 20,
                          from_vals: tuple = (0, 0, 0, 0),
                          to_vals: tuple = (0, 0, 0, 0),
                          label: str = ""):
        """Плавний перехід між двома станами стіків."""
        if label:
            print(f"  {label} ({duration:.1f} с) ...", flush=True)
        dt = duration / steps
        for i in range(steps + 1):
            t = i / steps
            vals = tuple(a + (b - a) * t for a, b in zip(from_vals, to_vals))
            self.set_sticks(*vals)
            time.sleep(dt)


# ── Тести ────────────────────────────────────────────────────────────────────


def test_axes(drone: DroneController):
    """Тест 1: Плавно рухає кожну вісь — перевірка що геймпад працює."""
    print("\n=== Тест 1: Перевірка осей ===")
    wait_for_game()

    axes = [
        ("Throttle Up",   (0.7, 0, 0, 0)),
        ("Throttle Down", (-0.7, 0, 0, 0)),
        ("Yaw Left",      (0, -0.7, 0, 0)),
        ("Yaw Right",     (0, 0.7, 0, 0)),
        ("Pitch Forward", (0, 0, -0.7, 0)),
        ("Pitch Back",    (0, 0, 0.7, 0)),
        ("Roll Left",     (0, 0, 0, -0.7)),
        ("Roll Right",    (0, 0, 0, 0.7)),
    ]

    for name, vals in axes:
        if keyboard.is_pressed("esc"):
            break
        drone.smooth_transition(0.5, from_vals=(0, 0, 0, 0), to_vals=vals, label=name)
        drone.smooth_transition(0.3, from_vals=vals, to_vals=(0, 0, 0, 0))
        time.sleep(0.2)

    drone.reset()
    print("Тест 1 завершено.\n")


def test_takeoff_hover_land(drone: DroneController):
    """Тест 2: Злетіти → повисіти 1 с → сісти."""
    print("\n=== Тест 2: Зліт → Зависання → Посадка ===")
    wait_for_game()

    # Крок 1: Плавно піднімаємо газ (зліт)
    drone.smooth_transition(
        duration=1.5,
        from_vals=(THROTTLE_IDLE, 0, 0, 0),
        to_vals=(THROTTLE_UP, 0, 0, 0),
        label="Зліт (плавний газ вгору)"
    )

    # Крок 2: Утримуємо газ для набору висоти
    drone.hold(duration=1.5, throttle=THROTTLE_UP, label="Набір висоти")

    # Крок 3: Зависання (газ у середину)
    drone.smooth_transition(
        duration=0.5,
        from_vals=(THROTTLE_UP, 0, 0, 0),
        to_vals=(THROTTLE_HOVER, 0, 0, 0),
        label="Перехід у зависання"
    )
    drone.hold(duration=1.0, throttle=THROTTLE_HOVER, label="Зависання (hover)")

    # Крок 4: Посадка (плавне зниження газу)
    drone.smooth_transition(
        duration=2.0,
        from_vals=(THROTTLE_HOVER, 0, 0, 0),
        to_vals=(THROTTLE_DOWN, 0, 0, 0),
        label="Посадка (плавне зниження)"
    )
    drone.hold(duration=1.0, throttle=THROTTLE_DOWN, label="Зниження")

    # Крок 5: Мотори на мінімум
    drone.smooth_transition(
        duration=0.5,
        from_vals=(THROTTLE_DOWN, 0, 0, 0),
        to_vals=(THROTTLE_IDLE, 0, 0, 0),
        label="Мотори вимкнено"
    )

    drone.reset()
    print("Тест 2 завершено.\n")


def test_square(drone: DroneController):
    """Тест 3: Злетіти і пролетіти квадратом."""
    print("\n=== Тест 3: Квадрат у повітрі ===")
    wait_for_game()

    # Зліт
    drone.smooth_transition(1.0, from_vals=(THROTTLE_IDLE, 0, 0, 0),
                            to_vals=(THROTTLE_UP, 0, 0, 0), label="Зліт")
    drone.hold(1.5, throttle=THROTTLE_UP, label="Набір висоти")
    drone.hold(0.5, throttle=THROTTLE_HOVER, label="Стабілізація")

    # Квадрат (hover + pitch/yaw)
    maneuvers = [
        ("Вперед",  (THROTTLE_HOVER, 0, PITCH_FORWARD, 0)),
        ("Вправо",  (THROTTLE_HOVER, YAW_RIGHT, 0, 0)),
        ("Назад",   (THROTTLE_HOVER, 0, PITCH_BACK, 0)),
        ("Вліво",   (THROTTLE_HOVER, YAW_LEFT, 0, 0)),
    ]

    for name, vals in maneuvers:
        if keyboard.is_pressed("esc"):
            break
        drone.hold(duration=1.5, throttle=vals[0], yaw=vals[1],
                   pitch=vals[2], roll=vals[3], label=name)
        drone.hold(duration=0.5, throttle=THROTTLE_HOVER, label="Пауза")

    # Посадка
    drone.smooth_transition(2.0, from_vals=(THROTTLE_HOVER, 0, 0, 0),
                            to_vals=(THROTTLE_IDLE, 0, 0, 0), label="Посадка")

    drone.reset()
    print("Тест 3 завершено.\n")


# ── Допоміжне ────────────────────────────────────────────────────────────────


def wait_for_game():
    """Дає 5 секунд переключитись на гру."""
    print("Переключись на Uncrashed протягом 5 секунд!")
    for i in range(5, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1)
    print("  Поїхали!\n")


# ── Меню ─────────────────────────────────────────────────────────────────────


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Тест віртуального Xbox-контролера для Uncrashed (Windows)  ║")
    print("║                                                             ║")
    print("║  Переконайся, що:                                          ║")
    print("║   1. ViGEmBus драйвер встановлено                          ║")
    print("║   2. Uncrashed запущено, дрон на землі                     ║")
    print("║   3. У грі обрано контролер (не клавіатуру)                ║")
    print("║                                                             ║")
    print("║  Натисни Esc для аварійної зупинки                         ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    drone = DroneController()
    print()

    while True:
        print("Обери тест:")
        print("  [1] Перевірка осей (рухає кожен стік)")
        print("  [2] Зліт → зависання → посадка")
        print("  [3] Квадрат у повітрі")
        print("  [q] Вийти\n")

        choice = input(">>> ").strip().lower()

        if choice == "1":
            test_axes(drone)
        elif choice == "2":
            test_takeoff_hover_land(drone)
        elif choice == "3":
            test_square(drone)
        elif choice == "q":
            break
        else:
            print("Невідомий вибір.\n")

    drone.reset()
    print("\n[INFO] Контролер скинуто. Завершено.")


if __name__ == "__main__":
    main()
