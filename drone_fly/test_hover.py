"""
Тест підльоту: плавно підняти газ до ~30%, потримати, плавно посадити.

Використання:
    1. На GX12: запустити vcpfly.lua, натиснути ENTER (ARM)
    2. На ПК:
       python test_hover.py --port COM5

    Ctrl+C = аварійна зупинка (газ → 0)

Залежності:
    pip install pyserial
"""

import argparse
import time
import sys


def main():
    parser = argparse.ArgumentParser(description="Тест підльоту ~43% газу")
    parser.add_argument("--port", type=str, default="COM5",
                        help="COM-порт GX12")
    parser.add_argument("--power", type=int, default=43,
                        help="Відсоток газу (default: 43)")
    parser.add_argument("--hover-time", type=float, default=1.0,
                        help="Час утримання в повітрі, сек (default: 1)")
    parser.add_argument("--ramp-time", type=float, default=1.5,
                        help="Час підйому/спуску газу, сек (default: 1.5)")
    args = parser.parse_args()

    import serial
    from rc_lua_serial import LuaVCPSender

    # Throttle: -1024 = мін (1000 мкс), 0 = центр (1500 мкс), 1024 = макс (2000 мкс)
    # 30% газу = -1024 + 0.30 * 2048 = -1024 + 614 = -410
    target_thr = int(-1024 + (args.power / 100.0) * 2048)
    idle_thr = -1024  # мінімум газу

    ramp_time = args.ramp_time
    hover_time = args.hover_time
    dt = 0.02  # 50 Hz

    print(f"[HOVER TEST] Порт: {args.port}")
    print(f"  Газ: {args.power}% (mixer value: {target_thr})")
    print(f"  Підйом: {ramp_time}с → Утримання: {hover_time}с → Спуск: {ramp_time * 2}с")
    print(f"  Загалом: {ramp_time + hover_time + ramp_time * 2:.1f}с")
    print()
    print("  ⚠️  Переконайся що vcpfly.lua запущено і натиснуто ENTER (ARMED)!")
    print("  ⚠️  Ctrl+C = аварійна зупинка")
    print()

    input("  Натисни ENTER щоб почати...")

    try:
        with LuaVCPSender(args.port) as sender:
            cycle = 0
            while True:
                cycle += 1
                print(f"\n  ═══ Цикл {cycle} ═══")

                # --- Фаза 1: Плавний підйом газу ---
                print("  [1/3] Підйом газу...")
                steps = int(ramp_time / dt)
                for i in range(steps):
                    progress = i / steps
                    thr = int(idle_thr + (target_thr - idle_thr) * progress)
                    sender.send(throttle=thr, roll=0, pitch=0, yaw=0)
                    pct = int((thr - idle_thr) / (target_thr - idle_thr) * args.power)
                    print(f"    Газ: {pct}%", end="\r")
                    time.sleep(dt)

                # --- Фаза 2: Утримання ---
                print(f"\n  [2/3] Утримання {hover_time}с на {args.power}%...")
                steps = int(hover_time / dt)
                for i in range(steps):
                    sender.send(throttle=target_thr, roll=0, pitch=0, yaw=0)
                    remaining = hover_time - i * dt
                    print(f"    Залишилось: {remaining:.1f}с", end="\r")
                    time.sleep(dt)

                # --- Фаза 3: Плавне зниження (2x повільніше ніж підйом) ---
                print(f"\n  [3/3] Зниження газу...")
                descent_time = ramp_time * 2
                steps = int(descent_time / dt)
                for i in range(steps):
                    progress = i / steps
                    thr = int(target_thr + (idle_thr - target_thr) * progress)
                    sender.send(throttle=thr, roll=0, pitch=0, yaw=0)
                    pct = max(0, int((thr - idle_thr) / (target_thr - idle_thr) * args.power))
                    print(f"    Газ: {pct}%", end="\r")
                    time.sleep(dt)

                # --- Пауза на землі ---
                sender.send(throttle=idle_thr, roll=0, pitch=0, yaw=0)
                print(f"\n  ✓ Цикл {cycle} завершено. Пауза 1с...")
                time.sleep(1.0)

    except serial.SerialException as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        # Аварійна зупинка
        print("\n\n  ⚠️  АВАРІЙНА ЗУПИНКА!")
        try:
            with LuaVCPSender(args.port) as sender:
                for _ in range(10):
                    sender.send(throttle=idle_thr, roll=0, pitch=0, yaw=0)
                    time.sleep(dt)
        except Exception:
            pass
        print("  Газ скинуто в 0")


if __name__ == "__main__":
    main()
