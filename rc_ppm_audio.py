"""
PPM-сигнал через Audio Jack (3.5mm) для тренерського порту Radiomaster TX12.

Генерує Pulse Position Modulation аудіо-сигнал, який пульт зчитує
як тренерський вхід через 3.5mm jack.

PPM Frame (стандарт):
    - Кількість каналів: 8 (перші 4 значущі: Roll, Pitch, Throttle, Yaw)
    - Тривалість імпульсу (pulse): 300 мкс
    - Значення каналу: 1000–2000 мкс (центр = 1500)
    - Sync pause: залишок до повного кадру (~22.5 мс)

Кабель: 3.5mm TRS → моно-сигнал на tip, GND на sleeve.
    Для Radiomaster TX12 тренерський порт — 3.5mm jack на задній панелі.

Залежності:
    pip install sounddevice numpy
"""

import numpy as np
import sounddevice as sd

# ── Налаштування PPM ────────────────────────────────────────────────────────

SAMPLE_RATE = 44100          # Гц — частота дискретизації аудіо
NUM_CHANNELS_PPM = 8         # Кількість каналів у PPM-кадрі
PULSE_WIDTH_US = 300         # Ширина імпульсу (мкс)
FRAME_LENGTH_US = 22500      # Загальна тривалість PPM-кадру (мкс)
PPM_POLARITY = 1.0           # 1.0 = позитивна полярність, -1.0 = негативна

# ── Генерація PPM ───────────────────────────────────────────────────────────


def _us_to_samples(us: float) -> int:
    """Перетворює мікросекунди на кількість семплів."""
    return int(round(us * SAMPLE_RATE / 1_000_000))


def generate_ppm_frame(channels: list[int]) -> np.ndarray:
    """
    Генерує один PPM-кадр як масив аудіо-семплів (float32, -1..1).

    Args:
        channels: список значень каналів (1000–2000 мкс). Довжина <= NUM_CHANNELS_PPM.
                  Якщо менше 8 каналів — решта заповнюється 1500 (центр).

    Returns:
        np.ndarray — аудіо-семпли одного PPM-кадру.
    """
    # Доповнюємо до 8 каналів
    ch = list(channels) + [1500] * (NUM_CHANNELS_PPM - len(channels))
    ch = ch[:NUM_CHANNELS_PPM]

    # Обмежуємо діапазон
    ch = [max(1000, min(2000, v)) for v in ch]

    pulse_samples = _us_to_samples(PULSE_WIDTH_US)
    total_frame_samples = _us_to_samples(FRAME_LENGTH_US)

    parts = []
    for value_us in ch:
        # Імпульс (HIGH)
        parts.append(np.full(pulse_samples, PPM_POLARITY, dtype=np.float32))
        # Пауза (LOW) = value - pulse_width
        gap_us = value_us - PULSE_WIDTH_US
        gap_samples = _us_to_samples(gap_us)
        parts.append(np.full(gap_samples, -PPM_POLARITY, dtype=np.float32))

    # Sync pulse
    parts.append(np.full(pulse_samples, PPM_POLARITY, dtype=np.float32))

    # Sync gap — залишок кадру
    used_samples = sum(p.shape[0] for p in parts)
    sync_gap_samples = total_frame_samples - used_samples
    if sync_gap_samples > 0:
        parts.append(np.full(sync_gap_samples, -PPM_POLARITY, dtype=np.float32))

    return np.concatenate(parts)


class PPMOutputStream:
    """
    Потоковий PPM-генератор — безперервно відтворює PPM-сигнал через аудіовихід.

    Використання:
        ppm = PPMOutputStream(device=None)  # None = дефолтний аудіовихід
        ppm.start()
        ppm.set_channels([1500, 1500, 1000, 1500])  # Roll, Pitch, Throttle, Yaw
        ...
        ppm.stop()
    """

    def __init__(self, device=None):
        """
        Args:
            device: індекс або назва аудіопристрою (None = системний дефолт).
                    Використай sd.query_devices() щоб побачити список.
        """
        self._device = device
        self._channels = [1500] * 4  # Roll, Pitch, Throttle, Yaw
        self._stream = None
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buf_pos = 0

    @property
    def channels(self) -> list[int]:
        return list(self._channels)

    def set_channels(self, channels: list[int]) -> None:
        """Оновити значення каналів (1000–2000 мкс). Thread-safe за рахунок GIL."""
        self._channels = [max(1000, min(2000, int(v))) for v in channels]

    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback для sounddevice — заповнює буфер PPM-семплами."""
        written = 0
        while written < frames:
            # Якщо буфер вичерпано — генеруємо новий кадр
            if self._buf_pos >= len(self._buffer):
                self._buffer = generate_ppm_frame(self._channels)
                self._buf_pos = 0

            chunk_size = min(frames - written, len(self._buffer) - self._buf_pos)
            outdata[written:written + chunk_size, 0] = \
                self._buffer[self._buf_pos:self._buf_pos + chunk_size]
            self._buf_pos += chunk_size
            written += chunk_size

    def start(self) -> None:
        """Запустити PPM-стрім."""
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self._device,
            callback=self._audio_callback,
            blocksize=1024,
        )
        self._stream.start()
        print(f"[PPM] Стрім запущено (SR={SAMPLE_RATE}, device={self._device})")

    def stop(self) -> None:
        """Зупинити PPM-стрім."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            print("[PPM] Стрім зупинено.")


# ── Швидкий тест ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    import math

    print("Доступні аудіопристрої:")
    print(sd.query_devices())
    print()

    ppm = PPMOutputStream(device=None)
    ppm.start()

    print("[TEST] Синусоїда на каналах Roll/Pitch. Ctrl+C для виходу.\n")
    try:
        t = 0.0
        while True:
            # Плавна синусоїда: 1000–2000, період ~4 секунди
            val = 1500 + 500 * math.sin(2 * math.pi * t / 4.0)
            ppm.set_channels([
                int(val),           # CH1 Roll
                int(val),           # CH2 Pitch
                1500,               # CH3 Throttle (середина)
                1500,               # CH4 Yaw (середина)
            ])
            print(f"  CH1={int(val)}  CH2={int(val)}  CH3=1500  CH4=1500", end="\r")
            time.sleep(0.02)
            t += 0.02
    except KeyboardInterrupt:
        pass

    ppm.stop()
