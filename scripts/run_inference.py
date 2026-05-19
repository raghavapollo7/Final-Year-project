import argparse
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.detector import RescueDetector
from src.preprocessing import preprocess_rubble, preprocess_underwater
from src.utils import load_yaml, open_video_source


def parse_args():
    parser = argparse.ArgumentParser(description="Run rescue detector on webcam or video.")
    parser.add_argument("--source", default="0", help="Webcam index or video file path.")
    parser.add_argument("--mode", choices=["underwater", "rubble"], default="underwater")
    parser.add_argument("--weights", default=None, help="Optional custom YOLO weights.")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--thermal", action="store_true", help="Enable thermal simulation in rubble mode.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml("config/app_config.yaml")
    if args.weights:
        weights = args.weights
    else:
        primary = (
            config["model"]["underwater_weights"]
            if args.mode == "underwater"
            else config["model"]["rubble_weights"]
        )
        weights = RescueDetector.choose_weights(primary, config["model"]["default_weights"])

    detector = RescueDetector(weights=weights, confidence=args.confidence, iou=float(config["model"]["iou"]))
    cap = open_video_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.mode == "underwater":
            processed = preprocess_underwater(frame)
        else:
            processed = preprocess_rubble(frame, thermal=args.thermal)

        annotated, detections = detector.detect(processed)
        cv2.putText(
            annotated,
            f"{args.mode.title()} Mode | Victims: {len(detections)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("AI Search and Rescue Drone", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
