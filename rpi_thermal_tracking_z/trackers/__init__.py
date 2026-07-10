"""Спільні модулі трекінгу для тепловізора MLX90642.

Винесена логіка з:
  * SORT_tracker_realtime.py        → trackers.sort
  * deepsort_temp_tracker_pi_001.py → trackers.deepsort_temp
  * deepsort_cnn_tracker_pi_002.py  → trackers.deepsort_cnn

Спільна сегментація (MRF + Graph-Cut) → trackers.segmentation
"""
from .segmentation import segment_frame, FRAME_H, FRAME_W
from .sort import SORTTrack, SORTTracker
from .deepsort_temp import (
    DeepSORTTempTrack,
    DeepSORTTempTracker,
    set_temp_range as set_temp_range_for_features,
    extract_temp_feature,
)

__all__ = [
    "FRAME_H",
    "FRAME_W",
    "segment_frame",
    "SORTTrack",
    "SORTTracker",
    "DeepSORTTempTrack",
    "DeepSORTTempTracker",
    "set_temp_range_for_features",
    "extract_temp_feature",
]
