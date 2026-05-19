from pathlib import Path

import cv2
import yaml


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_output_dirs():
    for folder in ["data/raw", "data/processed", "models", "outputs"]:
        Path(folder).mkdir(parents=True, exist_ok=True)


def open_video_source(source):
    """Open webcam index or video path."""
    try:
        source = int(source)
    except (TypeError, ValueError):
        pass
    return cv2.VideoCapture(source)


def bgr_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
