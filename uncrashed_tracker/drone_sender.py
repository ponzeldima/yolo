"""Відправка RC-команд на реальний дрон через USB-VCP LUA bridge.

Обгортка над LuaVCPSender з інтерфейсом, аналогічним VirtualGamepad:
stick values [-1, +1] → mixer values [-1024, +1024] → serial frame.

Ланцюг сигналу:
  Python (DroneSender) ─USB-VCP─► EdgeTX (LUA mixer) → ELRS/CRSF → Дрон
"""

import struct
import time

import serial


# ── Protocol Constants ───────────────────────────────────────────────────────

HEADER = bytes([0x55, 0xAA])
FRAME_SIZE = 11
VCP_BAUDRATE = 115200

MIXER_MIN = -1024
MIXER_MAX = 1024
WIRE_OFFSET = 1024


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _mixer_to_wire(val: int) -> int:
    return max(0, min(2048, val + WIRE_OFFSET))


def _build_frame(throttle: int, roll: int, pitch: int, yaw: int) -> bytes:
    channels = [
        _mixer_to_wire(throttle),
        _mixer_to_wire(roll),
        _mixer_to_wire(pitch),
        _mixer_to_wire(yaw),
    ]
    payload = struct.pack(">HHHH", *channels)
    xor = 0
    for b in payload:
        xor ^= b
    return HEADER + payload + bytes([xor])


# ── DroneSender ──────────────────────────────────────────────────────────────

class DroneSender:
    """Передача команд на реальний дрон через USB-VCP (LUA bridge).

    Має інтерфейс, сумісний з VirtualGamepad:
      init() / set_sticks(left_x, left_y, right_x, right_y) / reset() / close()

    Маппінг стіків (Mode 2):
      left_x  = Yaw       (-1 = ліво, +1 = право)
      left_y  = Throttle  (-1 = мінімум газу, +1 = максимум)
      right_x = Roll      (-1 = ліво, +1 = право)
      right_y = Pitch     (-1 = вперед, +1 = назад)
    """

    def __init__(self, port: str, baudrate: int = VCP_BAUDRATE):
        self._port = port
        self._baudrate = baudrate
        self._serial: serial.Serial | None = None

    def init(self):
        """Відкрити serial-порт і вивести в нейтраль."""
        self._serial = serial.Serial(
            port=self._port,
            baudrate=self._baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
        )
        print(f"[DRONE] Порт {self._port} відкрито ({self._baudrate} baud)")
        # Надсилаємо нейтраль (throttle = мінімум)
        self._send_mixer(throttle=MIXER_MIN, roll=0, pitch=0, yaw=0)

    def set_sticks(self, left_x: float, left_y: float,
                   right_x: float, right_y: float):
        """Встановити значення стіків ([-1, +1]) і надіслати фрейм.

        Маппінг (Mode 2):
          left_x  → Yaw
          left_y  → Throttle
          right_x → Roll
          right_y → Pitch
        """
        yaw = int(_clamp(left_x, -1.0, 1.0) * MIXER_MAX)
        throttle = int(_clamp(left_y, -1.0, 1.0) * MIXER_MAX)
        roll = int(_clamp(right_x, -1.0, 1.0) * MIXER_MAX)
        pitch = int(_clamp(right_y, -1.0, 1.0) * MIXER_MAX)

        self._send_mixer(throttle=throttle, roll=roll, pitch=pitch, yaw=yaw)

    def reset(self):
        """Скинути в безпечну нейтраль (throttle = мінімум)."""
        self._send_mixer(throttle=MIXER_MIN, roll=0, pitch=0, yaw=0)

    def close(self):
        """Безпечно закрити: скинути газ і закрити порт."""
        if self._serial and self._serial.is_open:
            try:
                # Надіслати кілька фреймів з нульовим газом
                for _ in range(5):
                    self._send_mixer(throttle=MIXER_MIN, roll=0, pitch=0, yaw=0)
                    time.sleep(0.02)
            except Exception:
                pass
            self._serial.close()
            print(f"[DRONE] Порт {self._port} закрито")

    def _send_mixer(self, throttle: int, roll: int, pitch: int, yaw: int):
        """Надіслати фрейм з mixer values (-1024..1024)."""
        if self._serial is None or not self._serial.is_open:
            return
        frame = _build_frame(throttle, roll, pitch, yaw)
        self._serial.write(frame)

    @property
    def is_open(self) -> bool:
        return self._serial is not None and self._serial.is_open
