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

# Телеметрія (Lua → Python)
TELEM_HEADER = bytes([0xAA, 0x55])
TELEM_FRAME_SIZE = 7
TELEM_OFFSET = 1800


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
        self._telem_buf = bytearray()
        self._telem_pitch: float | None = None
        self._telem_roll: float | None = None

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

    def read_telemetry(self) -> tuple[float | None, float | None]:
        """Non-blocking: читає телеметрію (pitch, roll) від Lua bridge.

        Returns:
            (pitch_deg, roll_deg) — градуси, або None якщо ще немає даних.
        """
        if self._serial is None or not self._serial.is_open:
            return self._telem_pitch, self._telem_roll

        avail = self._serial.in_waiting
        if avail > 0:
            self._telem_buf.extend(self._serial.read(avail))

        # Парсимо всі повні фрейми в буфері
        while len(self._telem_buf) >= TELEM_FRAME_SIZE:
            idx = self._telem_buf.find(TELEM_HEADER)
            if idx < 0:
                self._telem_buf.clear()
                break
            if idx > 0:
                del self._telem_buf[:idx]
            if len(self._telem_buf) < TELEM_FRAME_SIZE:
                break

            frame = self._telem_buf[:TELEM_FRAME_SIZE]
            xor = 0
            for b in frame[2:6]:
                xor ^= b
            if xor != frame[6]:
                del self._telem_buf[:1]
                continue

            pitch_raw = frame[2] * 256 + frame[3]
            roll_raw = frame[4] * 256 + frame[5]
            self._telem_pitch = (pitch_raw - TELEM_OFFSET) / 10.0
            self._telem_roll = (roll_raw - TELEM_OFFSET) / 10.0
            del self._telem_buf[:TELEM_FRAME_SIZE]

        return self._telem_pitch, self._telem_roll

    @property
    def telemetry_pitch(self) -> float | None:
        return self._telem_pitch

    @property
    def telemetry_roll(self) -> float | None:
        return self._telem_roll

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
