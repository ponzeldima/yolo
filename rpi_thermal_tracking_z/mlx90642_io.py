"""Мінімальне читання MLX90642 (та сама логіка, що у SORT_tracker_realtime.py).

Винесено окремим модулем, щоб усі три треккери керівника могли читати з сенсора,
а не лише з CSV.
"""
from __future__ import annotations

import numpy as np
from smbus2 import SMBus, i2c_msg

MLX90642_ADDR = 0x66
I2C_BUS       = 1
RAM_START     = 0x342C
CHUNK_WORDS   = 16
FRAME_H, FRAME_W = 24, 32


def _read_words(bus: SMBus, reg_addr: int, num_words: int) -> list[int]:
    wr = i2c_msg.write(MLX90642_ADDR, [(reg_addr >> 8) & 0xFF, reg_addr & 0xFF])
    rd = i2c_msg.read(MLX90642_ADDR, num_words * 2)
    bus.i2c_rdwr(wr, rd)
    return list(rd)


def _decode_temps(raw: list[int]) -> np.ndarray:
    vals = np.frombuffer(bytes(raw), dtype=">i2").astype(np.float32)
    return (vals / 50.0).reshape(FRAME_H, FRAME_W)


# Камера на дроні закріплена догори ногами — повертаємо кадр на 180°.
FLIP_180 = True


def read_frame(bus: SMBus) -> np.ndarray:
    """Один кадр (24×32 float32, °C). З FLIP_180=True повернутий на 180°."""
    raw: list[int] = []
    total_words = FRAME_H * FRAME_W   # 768
    for offset in range(0, total_words, CHUNK_WORDS):
        addr = RAM_START + offset * 2
        raw.extend(_read_words(bus, addr, CHUNK_WORDS))
    frame = _decode_temps(raw)
    if FLIP_180:
        frame = frame[::-1, ::-1]
    return np.ascontiguousarray(frame)


def open_bus() -> SMBus:
    return SMBus(I2C_BUS)
