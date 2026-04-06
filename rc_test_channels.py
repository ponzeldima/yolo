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


def main():
    parser = argparse.ArgumentParser(description="Тест RC-каналів (PPM або CRSF)")
    parser.add_argument("--method", choices=["ppm", "crsf"], required=True,
                        help="Метод передачі: ppm (audio jack) або crsf (USB serial)")
    parser.add_argument("--port", type=str, default="/dev/cu.usbmodem14201",
                        help="Serial-порт для CRSF (GX12 USB-C: /dev/cu.usbmodemXXXX)")
    parser.add_argument("--audio-device", type=int, default=None,
                        help="Індекс аудіопристрою для PPM (default: системний)")
    args = parser.parse_args()

    if args.method == "ppm":
        test_ppm(args.audio_device)
    else:
        test_crsf(args.port)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
