"""Джерела зображень (Strategy pattern).

Щоб замінити джерело — реалізуй BaseImageSource і передай у pipeline.
Наприклад: CameraSource, VideoFileSource тощо.
"""

from abc import ABC, abstractmethod

import numpy as np
import cv2
import dxcam
import win32gui

from . import config


class BaseImageSource(ABC):
    """Абстрактне джерело кадрів."""

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def get_frame(self) -> np.ndarray | None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @abstractmethod
    def get_region(self) -> tuple[int, int, int, int]:
        """Повертає (left, top, width, height) області захоплення."""
        ...

    @property
    @abstractmethod
    def width(self) -> int:
        ...

    @property
    @abstractmethod
    def height(self) -> int:
        ...


class DxcamSource(BaseImageSource):
    """Захоплення екрану через DXcam (для симулятора)."""

    def __init__(self, window_title: str, target_fps: int = 75):
        self._window_title = window_title
        self._target_fps = target_fps
        self._camera = None
        self._region = None
        self._left = 0
        self._top = 0
        self._width = 0
        self._height = 0

    @staticmethod
    def _find_window_rect(title_substring: str) -> tuple | None:
        result = []
        def _enum_cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                text = win32gui.GetWindowText(hwnd)
                if title_substring.lower() in text.lower():
                    result.append(win32gui.GetWindowRect(hwnd))
        win32gui.EnumWindows(_enum_cb, None)
        return result[0] if result else None

    def start(self) -> None:
        rect = self._find_window_rect(self._window_title)
        if rect is None:
            raise RuntimeError(f"Вікно '{self._window_title}' не знайдено. Запусти симулятор.")

        left, top, right, bottom = rect
        self._left = max(0, left)
        self._top = max(0, top)
        right = min(config.SCREEN_W, right)
        bottom = min(config.SCREEN_H, bottom)
        self._width = right - self._left
        self._height = bottom - self._top
        self._region = (self._left, self._top, right, bottom)

        self._camera = dxcam.create(output_color="BGR")
        self._camera.start(region=self._region, target_fps=self._target_fps)

    def get_frame(self) -> np.ndarray | None:
        return self._camera.get_latest_frame()

    def stop(self) -> None:
        if self._camera:
            self._camera.stop()
            del self._camera
            self._camera = None

    def get_region(self) -> tuple[int, int, int, int]:
        return (self._left, self._top, self._width, self._height)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def refresh_window(self) -> bool:
        """Оновлює позицію вікна. Повертає True якщо регіон змінився."""
        rect = self._find_window_rect(self._window_title)
        if rect is None:
            return False
        left, top, right, bottom = rect
        left, top = max(0, left), max(0, top)
        right = min(config.SCREEN_W, right)
        bottom = min(config.SCREEN_H, bottom)
        new_region = (left, top, right, bottom)
        if new_region != self._region:
            self._left, self._top = left, top
            self._width = right - left
            self._height = bottom - top
            self._region = new_region
            self._camera.stop()
            self._camera.start(region=self._region, target_fps=self._target_fps)
            return True
        return False


class CameraSource(BaseImageSource):
    """Захоплення з USB/вбудованої камери через OpenCV."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self._camera_index = camera_index
        self._req_width = width
        self._req_height = height
        self._cap: cv2.VideoCapture | None = None
        self._width = 0
        self._height = 0

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Камера {self._camera_index} не доступна.")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._req_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._req_height)
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAMERA] Відкрито камеру {self._camera_index}: {self._width}x{self._height}")

    def get_frame(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_region(self) -> tuple[int, int, int, int]:
        return (0, 0, self._width, self._height)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height
