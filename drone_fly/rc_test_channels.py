"""
Тестовий скрипт: вибір методу передачі RC-команд (PPM або CRSF)
з синусоїдальним тестовим сигналом на каналах.

Використання:
    python rc_test_channels.py --method ppm
    python rc_test_channels.py --method crsf --port /dev/cu.usbserial-0001

Залежності:
    PPM:  pip install sounddevice numpy
    CRSF: pip install pyserial
"""

import argparse
import math
import time
import sys


def test_ppm(audio_device: int | None) -> None:
    """Тестує PPM-вихід через аудіо."""
    from rc_ppm_audio import PPMOutputStream

    ppm = PPMOutputStream(device=audio_device)
    ppm.start()

    print("[TEST PPM] Синусоїда на CH1 (Roll) та CH2 (Pitch).")
    print("           CH3 (Throttle) = 1000→2000 повільно. CH4 (Yaw) = центр.")
    print("           Ctrl+C для виходу.\n")

    try:
        t = 0.0
        while True:
            # Roll/Pitch — швидка синусоїда (період 4 с)
            rp = 1500 + 500 * math.sin(2 * math.pi * t / 4.0)
            # Throttle — повільна синусоїда (період 8 с)
            thr = 1500 + 500 * math.sin(2 * math.pi * t / 8.0)
            # Yaw — ще повільніша (період 12 с)
            yaw = 1500 + 300 * math.sin(2 * math.pi * t / 12.0)

            ppm.set_channels([int(rp), int(rp), int(thr), int(yaw)])

            print(f"  CH1={int(rp):4d}  CH2={int(rp):4d}  "
                  f"CH3={int(thr):4d}  CH4={int(yaw):4d}", end="\r")

            time.sleep(0.02)  # 50 Hz update
            t += 0.02
    except KeyboardInterrupt:
        pass

    ppm.stop()


def test_crsf(port: str) -> None:
    """Тестує CRSF-вихід через serial."""
    import serial
    from rc_crsf_serial import CRSFSender

    print(f"[TEST CRSF] Підключення до {port}...")
    try:
        with CRSFSender(port) as sender:
            print("[TEST CRSF] Синусоїда на CH1 (Roll) та CH2 (Pitch).")
            print("            CH3 (Throttle) = повільна хвиля. CH4 (Yaw) = центр.")
            print("            Ctrl+C для виходу.\n")

            t = 0.0
            while True:
                rp = 1500 + 500 * math.sin(2 * math.pi * t / 4.0)
                thr = 1500 + 500 * math.sin(2 * math.pi * t / 8.0)
                yaw = 1500 + 300 * math.sin(2 * math.pi * t / 12.0)

                sender.send_channels([int(rp), int(rp), int(thr), int(yaw)])

                print(f"  CH1={int(rp):4d}  CH2={int(rp):4d}  "
                      f"CH3={int(thr):4d}  CH4={int(yaw):4d}", end="\r")

                time.sleep(0.004)  # ~250 Hz
                t += 0.004
    except serial.SerialException as e:
        print(f"[ERROR] {e}")
        print("        Перевір порт: ls /dev/cu.usb*")
        sys.exit(1)
    except KeyboardInterrupt:
        print()


def test_lua(port: str) -> None:
    """Тестує LUA VCP bridge."""
    import serial
    from rc_lua_serial import LuaVCPSender

    print(f"[TEST LUA] Підключення до {port}...")
    print("           USB-VCP Mode на пульті має бути: LUA")
    print("           LUA-скрипт: SCRIPTS/MIXES/vcp_bridge.lua\n")
    try:
        with LuaVCPSender(port) as sender:
            print("[TEST LUA] Синусоїда на каналах. Ctrl+C для виходу.\n")

            t = 0.0
            while True:
                # Roll/Pitch — синусоїда (період 4 с)
                rp = int(1024 * math.sin(2 * math.pi * t / 4.0))
                # Throttle — повільна (період 8 с), від -1024 до 1024
                thr = int(1024 * math.sin(2 * math.pi * t / 8.0))
                # Yaw — ще повільніша (період 12 с)
                yaw = int(600 * math.sin(2 * math.pi * t / 12.0))

                sender.send(throttle=thr, roll=rp, pitch=rp, yaw=yaw)

                # Відображення в мкс для зручності
                thr_us = 1500 + thr * 500 // 1024
                rp_us = 1500 + rp * 500 // 1024
                yaw_us = 1500 + yaw * 500 // 1024
                print(f"  Thr={thr:+5d} ({thr_us:4d}us)  "
                      f"Rol={rp:+5d} ({rp_us:4d}us)  "
                      f"Pit={rp:+5d} ({rp_us:4d}us)  "
                      f"Yaw={yaw:+5d} ({yaw_us:4d}us)", end="\r")

                time.sleep(0.02)  # 50 Hz
                t += 0.02
    except serial.SerialException as e:
        print(f"[ERROR] {e}")
        print("        Windows: перевір COM-порт у Device Manager")
        print("        macOS:   ls /dev/cu.usb*")
        sys.exit(1)
    except KeyboardInterrupt:
        print()


def main():
    parser = argparse.ArgumentParser(description="Тест RC-каналів (PPM, CRSF або LUA VCP)")
    parser.add_argument("--method", choices=["ppm", "crsf", "lua"], required=True,
                        help="Метод передачі: ppm (audio) / crsf (USB serial) / lua (VCP bridge)")
    parser.add_argument("--port", type=str, default="COM5",
                        help="Serial-порт (Windows: COM5, macOS: /dev/cu.usbmodemXXXX)")
    parser.add_argument("--audio-device", type=int, default=None,
                        help="Індекс аудіопристрою для PPM (default: системний)")
    args = parser.parse_args()

    if args.method == "ppm":
        test_ppm(args.audio_device)
    elif args.method == "crsf":
        test_crsf(args.port)
    else:
        test_lua(args.port)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
