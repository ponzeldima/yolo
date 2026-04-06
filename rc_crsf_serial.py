"""
CRSF (Crossfire) протокол через USB-Serial для тренерського порту.

Формує CRSF RC Channels Packed frame і відправляє через serial-порт
на Radiomaster GX12 (через USB-C) або будь-який ELRS/CRSF-сумісний пульт.

Для GX12: підключення USB-C кабелем, обрати режим "USB Serial (VCP)".
Для пультів з 3.5mm jack: через USB→UART адаптер 3.3V.

CRSF Frame structure:
    [Device Address] [Frame Length] [Frame Type] [Payload...] [CRC8]

RC Channels Packed (Type 0x16):
    - 16 каналів × 11 біт = 22 байти payload
    - Діапазон значень: 172..1811 (CRSF internal), мапиться на 1000..2000 мкс

Залежності:
    pip install pyserial
"""

import struct
import time
import serial

# ── CRSF Constants ───────────────────────────────────────────────────────────

CRSF_SYNC = 0xC8                    # Device address (Flight Controller / крайній)
CRSF_FRAMETYPE_RC_CHANNELS = 0x16   # RC Channels Packed
CRSF_NUM_CHANNELS = 16
CRSF_CHANNEL_BITS = 11
CRSF_BAUDRATE = 420000

# Діапазон CRSF internal value
CRSF_CHANNEL_MIN = 172
CRSF_CHANNEL_MID = 992
CRSF_CHANNEL_MAX = 1811

# CRC8 таблиця (поліном 0xD5 — стандарт CRSF)
_CRC8_TABLE = None


def _build_crc8_table() -> list[int]:
    """Генерує CRC8 lookup table з поліномом 0xD5."""
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0xD5) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
        table.append(crc)
    return table


def _crc8(data: bytes) -> int:
    """Обчислює CRC8 для CRSF-пакета (поліном 0xD5)."""
    global _CRC8_TABLE
    if _CRC8_TABLE is None:
        _CRC8_TABLE = _build_crc8_table()
    crc = 0
    for b in data:
        crc = _CRC8_TABLE[crc ^ b]
    return crc


# ── Конвертація каналів ──────────────────────────────────────────────────────


def us_to_crsf(us: int) -> int:
    """
    Конвертує значення каналу з мікросекунд (1000–2000) у CRSF internal (172–1811).
    """
    us = max(1000, min(2000, us))
    return int(CRSF_CHANNEL_MIN + (us - 1000) * (CRSF_CHANNEL_MAX - CRSF_CHANNEL_MIN) / 1000)


def crsf_to_us(crsf_val: int) -> int:
    """
    Конвертує CRSF internal (172–1811) у мікросекунди (1000–2000).
    """
    return int(1000 + (crsf_val - CRSF_CHANNEL_MIN) * 1000 / (CRSF_CHANNEL_MAX - CRSF_CHANNEL_MIN))


# ── Побудова пакета ──────────────────────────────────────────────────────────


def pack_channels(channels: list[int]) -> bytes:
    """
    Пакує 16 каналів (CRSF internal values, 11 біт кожен) у 22 байти payload.

    Args:
        channels: список з 16 значень (172–1811). Якщо менше — доповнюється CRSF_CHANNEL_MID.
    """
    ch = list(channels) + [CRSF_CHANNEL_MID] * (CRSF_NUM_CHANNELS - len(channels))
    ch = ch[:CRSF_NUM_CHANNELS]

    # Pack 16 × 11-bit values into a bit stream
    bit_stream = 0
    for i, val in enumerate(ch):
        val = max(CRSF_CHANNEL_MIN, min(CRSF_CHANNEL_MAX, val))
        bit_stream |= (val & 0x7FF) << (i * CRSF_CHANNEL_BITS)

    # Extract 22 bytes
    payload = bytearray(22)
    for i in range(22):
        payload[i] = (bit_stream >> (i * 8)) & 0xFF

    return bytes(payload)


def build_rc_channels_frame(channels_us: list[int]) -> bytes:
    """
    Будує повний CRSF RC Channels Packed frame.

    Args:
        channels_us: список значень каналів у мікросекундах (1000–2000).
                     Перші 4 = Roll, Pitch, Throttle, Yaw. Решта — 1500.

    Returns:
        bytes — готовий CRSF-пакет для відправки в UART.
    """
    # Конвертуємо мікросекунди → CRSF internal
    crsf_values = [us_to_crsf(v) for v in channels_us]
    # Доповнюємо до 16 каналів
    crsf_values += [CRSF_CHANNEL_MID] * (CRSF_NUM_CHANNELS - len(crsf_values))
    crsf_values = crsf_values[:CRSF_NUM_CHANNELS]

    payload = pack_channels(crsf_values)

    # Frame: [type] [payload] — для CRC
    frame_body = bytes([CRSF_FRAMETYPE_RC_CHANNELS]) + payload
    crc = _crc8(frame_body)

    # Повний пакет: [sync] [length] [type] [payload] [crc]
    frame_length = len(frame_body) + 1  # +1 для CRC
    packet = bytes([CRSF_SYNC, frame_length]) + frame_body + bytes([crc])
    return packet


# ── CRSF Serial sender ──────────────────────────────────────────────────────


class CRSFSender:
    """
    Відправляє CRSF RC Channels через serial-порт.

    Використання:
        # GX12 через USB-C:
        sender = CRSFSender("/dev/cu.usbmodem14201")
        # Пульти з 3.5mm jack через USB→UART адаптер:
        # sender = CRSFSender("/dev/cu.usbserial-0001")
        sender.open()
        sender.send_channels([1500, 1500, 1000, 1500])  # Roll, Pitch, Thr, Yaw
        sender.close()
    """

    def __init__(self, port: str, baudrate: int = CRSF_BAUDRATE):
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
        print(f"[CRSF] Порт {self._port} відкрито ({self._baudrate} baud)")

    def close(self) -> None:
        """Закрити serial-порт."""
        if self._serial and self._serial.is_open:
            self._serial.close()
            print(f"[CRSF] Порт {self._port} закрито")

    def send_channels(self, channels_us: list[int]) -> None:
        """
        Відправити RC Channels frame.

        Args:
            channels_us: значення каналів у мкс (1000–2000). Мінімум 4 (Roll, Pitch, Thr, Yaw).
        """
        if self._serial is None or not self._serial.is_open:
            raise RuntimeError("Serial-порт не відкрито. Виклич .open() спочатку.")
        packet = build_rc_channels_frame(channels_us)
        self._serial.write(packet)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


# ── Утиліта: список доступних serial-портів ──────────────────────────────────


def list_serial_ports() -> list[str]:
    """Повертає список доступних serial-портів на macOS."""
    import serial.tools.list_ports
    return [p.device for p in serial.tools.list_ports.comports()]


# ── Швидкий тест ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math

    print("Доступні serial-порти:")
    ports = list_serial_ports()
    for p in ports:
        print(f"  {p}")
    if not ports:
        print("  (жодного не знайдено)")
    print()

    # ── Зміни на свій порт ──
    # GX12 через USB-C — зазвичай /dev/cu.usbmodemXXXX
    # Пульти з 3.5mm jack через USB→UART — /dev/cu.usbserial-XXXX
    SERIAL_PORT = "/dev/cu.usbmodem14201"  # ← Зміни на свій! (ls /dev/cu.usb*)

    print(f"[TEST] Спроба підключення до {SERIAL_PORT}...")
    try:
        with CRSFSender(SERIAL_PORT) as sender:
            print("[TEST] Синусоїда на каналах Roll/Pitch. Ctrl+C для виходу.\n")
            t = 0.0
            while True:
                val = 1500 + 500 * math.sin(2 * math.pi * t / 4.0)
                channels = [
                    int(val),   # CH1 Roll
                    int(val),   # CH2 Pitch
                    1500,       # CH3 Throttle
                    1500,       # CH4 Yaw
                ]
                sender.send_channels(channels)
                print(f"  CH1={int(val)}  CH2={int(val)}  CH3=1500  CH4=1500", end="\r")
                time.sleep(0.004)  # ~250 Hz — стандартна частота CRSF
                t += 0.004
    except serial.SerialException as e:
        print(f"[ERROR] Не вдалося відкрити порт: {e}")
        print("        Перевір кабель та назву порту (ls /dev/cu.usb*)")
    except KeyboardInterrupt:
        print("\n[TEST] Зупинено.")
