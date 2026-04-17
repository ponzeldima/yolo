"""Бекенди відображення (Strategy pattern).

Щоб замінити спосіб виводу — реалізуй BaseDisplay.
"""

from abc import ABC, abstractmethod

import numpy as np
import cv2
import win32gui

from .overlay import GameOverlay


class BaseDisplay(ABC):
    """Абстрактний бекенд відображення."""

    @abstractmethod
    def show(self, frame: np.ndarray) -> None:
        ...

    @abstractmethod
    def should_quit(self) -> bool:
        ...

    @abstractmethod
    def destroy(self) -> None:
        ...

    def reposition(self, x: int, y: int, w: int, h: int) -> None:
        """Оновити позицію/розмір (за замовчуванням нічого не робить)."""


class OverlayDisplay(BaseDisplay):
    """Прозорий overlay поверх гри (UpdateLayeredWindow)."""

    def __init__(self, x: int, y: int, w: int, h: int):
        self._overlay = GameOverlay(x, y, w, h)
        print(f"[INFO] Overlay поверх гри ({w}x{h})")

    def show(self, frame: np.ndarray) -> None:
        self._overlay.update_async(frame)
        win32gui.PumpWaitingMessages()

    def should_quit(self) -> bool:
        return False  # quit handled via keyboard hook

    def reposition(self, x: int, y: int, w: int, h: int) -> None:
        self._overlay.reposition(x, y, w, h)

    def destroy(self) -> None:
        self._overlay.wait_done()
        self._overlay.destroy()


class WindowDisplay(BaseDisplay):
    """Окреме вікно OpenCV."""

    WINDOW_NAME = "Drone Visual Aim"

    def __init__(self):
        self._quit = False

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow(self.WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._quit = True

    def should_quit(self) -> bool:
        return self._quit

    def destroy(self) -> None:
        cv2.destroyAllWindows()
