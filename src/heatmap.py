import cv2
import numpy as np


class DetectionHeatmap:
    """Accumulates detection regions so the dashboard can show likely victim areas."""

    def __init__(self, decay: float = 0.92):
        self.decay = decay
        self.map = None

    def update(self, frame_shape, detections):
        height, width = frame_shape[:2]
        if self.map is None or self.map.shape != (height, width):
            self.map = np.zeros((height, width), dtype=np.float32)

        self.map *= self.decay
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            self.map[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] += detection["confidence"]

        normalized = cv2.normalize(self.map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    @staticmethod
    def overlay(frame, heatmap, alpha: float = 0.35):
        return cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
