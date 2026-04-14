"""
USB-VCP LUA Bridge — передача RC-команд через EdgeTX LUA mixer скрипт.

Замість CRSF-протоколу (trainer input), використовується простий бінарний
протокол, який LUA-скрипт на апаратурі читає через serialRead().

Принцип:
    PC (Python) ──USB-VCP──► EdgeTX (LUA mixer) → ELRS/CRSF → Дрон

Режим USB на пульті: LUA (SYS → Hardware → USB-VCP Mode → LUA)
LUA-скрипт: SCRIPTS/MIXES/vcp_bridge.lua (активувати в MDL → Mixes)

Протокол (11 байт на фрейм):
    [0x55] [0xAA] [thr_h thr_l] [rol_h rol_l] [pit_h pit_l] [yaw_h yaw_l] [xor]

    - Кожен канал: uint16 big-endian, значення = (mixer_value + 1024)
      де mixer_value: -1024..1024 → wire_value: 0..2048
    - XOR: побайтовий XOR байтів 2..9 (8 байт каналів)
    - Частота: 50 Hz (кожні 20 мс)

Залежності:
    pip install pyserial
"""

import struct
import time
import serial

# ── Protocol Constants ───────────────────────────────────────────────────────

HEADER = bytes([0x55, 0xAA])
FRAME_SIZE = 11
VCP_BAUDRATE = 115200  # USB-VCP ігнорує baudrate, але pyserial вимагає

# Mixer range in EdgeTX
MIXER_MIN = -1024
MIXER_MAX = 1024
MIXER_MID = 0

# Offset for wire encoding: -1024..1024 → 0..2048
WIRE_OFFSET = 1024


# ── Конвертація ──────────────────────────────────────────────────────────────

def us_to_mixer(us: int) -> int:
    """
    Конвертує мікросекунди (1000–2000) у mixer value (-1024..1024).
    1000 мкс → -1024, 1500 мкс → 0, 2000 мкс → 1024
    """
    us = max(1000, min(2000, us))
    return int((us - 1500) * 1024 / 500)


def mixer_to_us(mixer_val: int) -> int:
    """
    Конвертує mixer value (-1024..1024) у мікросекунди (1000–2000).
    """
    mixer_val = max(MIXER_MIN, min(MIXER_MAX, mixer_val))
    return int(1500 + mixer_val * 500 / 1024)


def mixer_to_wire(val: int) -> int:
    """Mixer value (-1024..1024) → wire value (0..2048)."""
    return max(0, min(2048, val + WIRE_OFFSET))


def wire_to_mixer(wire: int) -> int:
    """Wire value (0..2048) → mixer value (-1024..1024)."""
    return max(MIXER_MIN, min(MIXER_MAX, wire - WIRE_OFFSET))


# ── Побудова фрейму ──────────────────────────────────────────────────────────

def build_frame(throttle: int, roll: int, pitch: int, yaw: int) -> bytes:
    """
    Будує 11-байтний фрейм для відправки через VCP.

    Args:
        throttle, roll, pitch, yaw: mixer values (-1024..1024)

    Returns:
        bytes — 11-байтний фрейм [header(2) + channels(8) + xor(1)]
    """
    channels = [
        mixer_to_wire(throttle),
        mixer_to_wire(roll),
        mixer_to_wire(pitch),
        mixer_to_wire(yaw),
    ]

    # Pack 4 channels as big-endian uint16
    payload = struct.pack(">HHHH", *channels)

    # XOR checksum of channel bytes
    xor = 0
    for b in payload:
        xor ^= b

    return HEADER + payload + bytes([xor])


def build_frame_us(channels_us: list[int]) -> bytes:
    """
    Будує фрейм з каналів у мікросекундах.

    Args:
        channels_us: [throttle, roll, pitch, yaw] у мкс (1000–2000).
                     Якщо менше 4 — доповнюється 1500 (центр).
    """
    ch = list(channels_us) + [1500] * (4 - len(channels_us))
    ch = ch[:4]
    return build_frame(
        us_to_mixer(ch[0]),
        us_to_mixer(ch[1]),
        us_to_mixer(ch[2]),
        us_to_mixer(ch[3]),
    )


# ── LUA VCP Sender ──────────────────────────────────────────────────────────

class LuaVCPSender:
    """
    Відправляє RC-дані через USB-VCP для LUA mixer скрипта на EdgeTX.

    Використання:
        sender = LuaVCPSender("COM5")  # Windows
        sender = LuaVCPSender("/dev/cu.usbmodem14201")  # macOS
        sender.open()
        sender.send(throttle=0, roll=0, pitch=0, yaw=0)  # нейтраль
        sender.send_us([1500, 1500, 1000, 1500])  # мікросекунди
        sender.close()
    """

    def __init__(self, port: str, baudrate: int = VCP_BAUDRATE):
        self._port = port
        self._baudrate = baudrate
        self._serial: serial.Serial | None = None

    def open(self) -> None:
        """Відкрити serial-порт."""
        self._serial = serial.Serial(
            port=self._port,
            baudrate=self._baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
        )
        print(f"[LUA-VCP] Порт {self._port} відкрито ({self._baudrate} baud)")

    def close(self) -> None:
        """Закрити serial-порт."""
        if self._serial and self._serial.is_open:
            self._serial.close()
            print(f"[LUA-VCP] Порт {self._port} закрито")

    def send(self, throttle: int = 0, roll: int = 0,
             pitch: int = 0, yaw: int = 0) -> None:
        """
        Відправити канали (mixer values: -1024..1024).
        0 = центр (нейтраль), -1024 = мін, 1024 = макс.
        """
        if self._serial is None or not self._serial.is_open:
            raise RuntimeError("Serial-порт не відкрито. Виклич .open() спочатку.")
        frame = build_frame(throttle, roll, pitch, yaw)
        self._serial.write(frame)

    def send_us(self, channels_us: list[int]) -> None:
        """
        Відправити канали у мікросекундах (1000–2000).

        Args:
            channels_us: [throttle, roll, pitch, yaw]
        """
        if self._serial is None or not self._serial.is_open:
            raise RuntimeError("Serial-порт не відкрито. Виклич .open() спочатку.")
        frame = build_frame_us(channels_us)
        self._serial.write(frame)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
