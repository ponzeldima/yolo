"""Контролери вводу/виводу: фізичний джойстик + віртуальний геймпад."""

import time

import pygame
import vgamepad as vg

from . import config


class PhysicalJoystick:
    """Читання фізичного контролера через pygame."""

    def __init__(self):
        self._joy = None

    def init(self) -> bool:
        pygame.init()
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        if count == 0:
            print("[WARN] Фізичний контролер не знайдено! MANUAL режим не працюватиме.")
            return False
        # Вибираємо фізичний контролер (не віртуальний Xbox)
        for i in range(count):
            joy = pygame.joystick.Joystick(i)
            joy.init()
            name = joy.get_name().lower()
            # Пропускаємо віртуальні геймпади (ViGEmBus)
            if "xbox" in name or "x-box" in name or "vigem" in name or "virtual" in name:
                print(f"[INFO] Пропускаю віртуальний геймпад: [{i}] {joy.get_name()}")
                continue
            self._joy = joy
            print(f"[INFO] Фізичний контролер: [{i}] {joy.get_name()} ({joy.get_numaxes()} осей)")
            return True
        print("[WARN] Фізичний контролер не знайдено (всі джойстіки — віртуальні)!")
        return False

    def read(self) -> tuple[float, float, float, float]:
        """Повертає (roll, pitch, throttle, yaw) в [-1, +1]."""
        if self._joy is None:
            return (0.0, 0.0, -1.0, 0.0)

        pygame.event.pump()

        def read_axis(idx: int, invert: bool) -> float:
            if idx < self._joy.get_numaxes():
                val = self._joy.get_axis(idx)
                return -val if invert else val
            return 0.0

        roll = read_axis(config.PHYS_AXIS_ROLL, config.INVERT_ROLL)
        pitch = read_axis(config.PHYS_AXIS_PITCH, config.INVERT_PITCH)
        throttle = read_axis(config.PHYS_AXIS_THROTTLE, config.INVERT_THROTTLE)
        yaw = read_axis(config.PHYS_AXIS_YAW, config.INVERT_YAW)

        return (roll, pitch, throttle, yaw)

    def quit(self):
        pygame.quit()


class VirtualGamepad:
    """Віртуальний Xbox 360 контролер через ViGEmBus."""

    def __init__(self):
        self._gamepad = None

    def init(self):
        print("[INFO] Створюю віртуальний Xbox 360 контролер...")
        self._gamepad = vg.VX360Gamepad()
        self._gamepad.left_joystick_float(x_value_float=0.0, y_value_float=-1.0)
        self._gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self._gamepad.update()
        time.sleep(0.5)

    def set_sticks(self, left_x: float, left_y: float, right_x: float, right_y: float):
        """Встановлює значення стіків (clamp до [-1, +1])."""
        self._gamepad.left_joystick_float(
            x_value_float=max(-1.0, min(1.0, left_x)),
            y_value_float=max(-1.0, min(1.0, left_y)),
        )
        self._gamepad.right_joystick_float(
            x_value_float=max(-1.0, min(1.0, right_x)),
            y_value_float=max(-1.0, min(1.0, right_y)),
        )
        self._gamepad.update()

    def reset(self):
        """Скидає стіки в нейтраль."""
        self._gamepad.left_joystick_float(x_value_float=0.0, y_value_float=-1.0)
        self._gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self._gamepad.update()
