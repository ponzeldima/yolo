"""Трекери (Strategy pattern).

Щоб замінити трекер — реалізуй BaseTracker і передай у pipeline.

Доступні трекери:
    YoloByteTracker   — YOLOv8 + ByteTrack (за замовчуванням)
    YoloBotSortTracker — YOLOv8 + BoT-SORT (з motion compensation)
    YoloReIDTracker    — YOLOv8 + BoT-SORT + ReID (appearance matching)
    RtDetrTracker      — RT-DETR (transformer) + ByteTrack
"""

from abc import ABC, abstractmethod

import numpy as np
from ultralytics import YOLO, RTDETR

from .models import Detection
from . import config


class BaseTracker(ABC):
    """Абстрактний трекер: приймає кадр, повертає список детекцій."""

    @abstractmethod
    def track(self, frame: np.ndarray) -> list[Detection]:
        ...


def _parse_results(results, conf_threshold: float) -> list[Detection]:
    """Спільний парсинг результатів ultralytics в список Detection."""
    detections = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf >= conf_threshold:
            track_id = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                confidence=conf, track_id=track_id,
            ))
    return detections


# ── 1. YOLOv8 + ByteTrack (оригінальний трекер) ────────────────────────────

class YoloByteTracker(BaseTracker):
    """YOLOv8 + ByteTrack — швидкий IoU-based трекінг.

    Плюси: швидкий, простий, мінімум параметрів.
    Мінуси: може плутати ID при перекритті об'єктів.
    """

    def __init__(
        self,
        model_path: str = config.MODEL_PATH,
        imgsz: int = config.IMGSZ,
        conf: float = config.CONFIDENCE_THRESHOLD,
        classes: list[int] | None = None,
        device: str = "cuda",
        half: bool = True,
    ):
        self._model = YOLO(model_path)
        self._imgsz = imgsz
        self._conf = conf
        self._classes = classes or [config.CAR_CLASS_ID]
        self._device = device
        self._half = half

    def track(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.track(
            frame, verbose=False, conf=self._conf,
            classes=self._classes, device=self._device,
            persist=True, imgsz=self._imgsz, half=self._half,
            tracker="bytetrack.yaml",
        )
        return _parse_results(results, self._conf)


# Зворотна сумісність: старий YoloTracker = YoloByteTracker
YoloTracker = YoloByteTracker


# ── 2. YOLOv8 + BoT-SORT (motion compensation) ─────────────────────────────

class YoloBotSortTracker(BaseTracker):
    """YOLOv8 + BoT-SORT — трекінг з компенсацією руху камери (GMC).

    Плюси: краще при русі камери (дрон!), sparseOptFlow GMC.
    Мінуси: трохи повільніше за ByteTrack.
    """

    def __init__(
        self,
        model_path: str = config.MODEL_PATH,
        imgsz: int = config.IMGSZ,
        conf: float = config.CONFIDENCE_THRESHOLD,
        classes: list[int] | None = None,
        device: str = "cuda",
        half: bool = True,
    ):
        self._model = YOLO(model_path)
        self._imgsz = imgsz
        self._conf = conf
        self._classes = classes or [config.CAR_CLASS_ID]
        self._device = device
        self._half = half

    def track(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.track(
            frame, verbose=False, conf=self._conf,
            classes=self._classes, device=self._device,
            persist=True, imgsz=self._imgsz, half=self._half,
            tracker="botsort.yaml",
        )
        return _parse_results(results, self._conf)


# ── 3. YOLOv8 + BoT-SORT + ReID (appearance-based) ────────────────────────

class YoloReIDTracker(BaseTracker):
    """YOLOv8 + BoT-SORT + ReID — трекінг з appearance matching.

    Еквівалент StrongSORT: використовує ReID модель для розпізнавання
    об'єктів за зовнішнім виглядом, а не тільки за IoU/рухом.

    Плюси: найкраще утримання ID при перекритті та різких маневрах.
    Мінуси: найповільніший (ReID модель), потребує більше VRAM.
    """

    def __init__(
        self,
        model_path: str = config.MODEL_PATH,
        imgsz: int = config.IMGSZ,
        conf: float = config.CONFIDENCE_THRESHOLD,
        classes: list[int] | None = None,
        device: str = "cuda",
        half: bool = True,
        reid_model: str = "auto",
    ):
        self._model = YOLO(model_path)
        self._imgsz = imgsz
        self._conf = conf
        self._classes = classes or [config.CAR_CLASS_ID]
        self._device = device
        self._half = half
        self._reid_model = reid_model

        # Створюємо кастомний конфіг BoT-SORT з ReID
        import tempfile, os, yaml
        botsort_reid = {
            "tracker_type": "botsort",
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.1,
            "new_track_thresh": 0.25,
            "track_buffer": 50,
            "match_thresh": 0.8,
            "fuse_score": True,
            "gmc_method": "sparseOptFlow",
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.8,
            "with_reid": True,
            "model": reid_model,
        }
        self._tracker_cfg = os.path.join(tempfile.gettempdir(), "botsort_reid.yaml")
        with open(self._tracker_cfg, "w") as f:
            yaml.dump(botsort_reid, f)

    def track(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.track(
            frame, verbose=False, conf=self._conf,
            classes=self._classes, device=self._device,
            persist=True, imgsz=self._imgsz, half=self._half,
            tracker=self._tracker_cfg,
        )
        return _parse_results(results, self._conf)


# ── 4. RT-DETR + ByteTrack (transformer detector) ──────────────────────────

class RtDetrTracker(BaseTracker):
    """RT-DETR (Real-Time DEtection TRansformer) + ByteTrack.

    Замість YOLO (CNN) використовує transformer-based детектор.
    RT-DETR не потребує NMS — end-to-end детекція.

    Плюси: transformer архітектура, краще при складних сценах,
           менше дублів (без NMS).
    Мінуси: повільніший за YOLOv8n, більший розмір моделі.

    Доступні моделі: rtdetr-l.pt, rtdetr-x.pt
    """

    def __init__(
        self,
        model_path: str = "rtdetr-l.pt",
        imgsz: int = 640,
        conf: float = config.CONFIDENCE_THRESHOLD,
        classes: list[int] | None = None,
        device: str = "cuda",
        half: bool = True,
    ):
        self._model = RTDETR(model_path)
        self._imgsz = imgsz
        self._conf = conf
        self._classes = classes or [config.CAR_CLASS_ID]
        self._device = device
        self._half = half

    def track(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.track(
            frame, verbose=False, conf=self._conf,
            classes=self._classes, device=self._device,
            persist=True, imgsz=self._imgsz, half=self._half,
            tracker="bytetrack.yaml",
        )
        return _parse_results(results, self._conf)
