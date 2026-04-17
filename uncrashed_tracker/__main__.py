"""Entry point: python -m uncrashed_tracker"""

from . import config
from .image_source import DxcamSource, CameraSource
from .tracker import (
    YoloByteTracker, YoloBotSortTracker, YoloReIDTracker, RtDetrTracker,
)
from .display import OverlayDisplay, WindowDisplay
from .pipeline import DroneAimPipeline
from .camera_pipeline import CameraTrackingPipeline
from .drone_pipeline import DroneTrackingPipeline


def main():
    tracker = YoloBotSortTracker()

    if config.INPUT_MODE == "drone":
        # ── Дрон: камера + трекінг + джойстік/автопілот → реальний дрон ──
        source = CameraSource(config.CAMERA_INDEX, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        source.start()
        pipeline = DroneTrackingPipeline(source, tracker, config.DRONE_PORT)

    elif config.INPUT_MODE == "camera":
        # ── Камера: детекція + відображення у вікні ──
        source = CameraSource(config.CAMERA_INDEX, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        source.start()
        pipeline = CameraTrackingPipeline(source, tracker)

    else:
        # ── Симулятор: повний конвеєр з автопілотом ──
        source = DxcamSource(config.WINDOW_TITLE)
        source.start()
        left, top, width, height = source.get_region()

        if config.DISPLAY_MODE == "overlay":
            display = OverlayDisplay(left, top, width, height)
        else:
            display = WindowDisplay()

        pipeline = DroneAimPipeline(source, tracker, display)

    pipeline.run()


if __name__ == "__main__":
    main()
