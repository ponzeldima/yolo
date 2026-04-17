"""Загальні моделі даних."""

from dataclasses import dataclass


@dataclass
class Detection:
    """Виявлена ціль."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    track_id: int = -1

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1
