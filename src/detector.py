from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


PERSON_CLASS_ID = 0


class RescueDetector:
    """YOLOv8 person detector wrapper used by the dashboard and CLI."""

    def __init__(self, weights: str, confidence: float = 0.35, iou: float = 0.45):
        self.weights = weights
        self.confidence = confidence
        self.iou = iou
        self.model = YOLO(weights)

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Run YOLO inference and draw rescue-focused annotations."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
            classes=[PERSON_CLASS_ID],
        )

        detections: List[Dict] = []
        annotated = frame.copy()
        if not results:
            return annotated, detections

        boxes = results[0].boxes
        if boxes is None:
            return annotated, detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            detections.append(
                {
                    "class": "person",
                    "confidence": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )
            self._draw_detection(annotated, x1, y1, x2, y2, conf)

        return annotated, detections

    @staticmethod
    def choose_weights(primary: str, fallback: str) -> str:
        """Use a trained project model when available; otherwise use YOLOv8 pretrained weights."""
        return primary if Path(primary).exists() else fallback

    @staticmethod
    def _draw_detection(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, conf: float) -> None:
        color = (0, 255, 80) if conf >= 0.5 else (0, 180, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"Human detected {conf * 100:.1f}%"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y1, label_size[1] + 10)
        cv2.rectangle(
            frame,
            (x1, label_y - label_size[1] - 10),
            (x1 + label_size[0] + 8, label_y + 4),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
