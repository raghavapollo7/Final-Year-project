import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for rescue human detection.")
    parser.add_argument("--data", required=True, help="Path to YOLO dataset YAML.")
    parser.add_argument("--weights", default="yolov8n.pt", help="Base YOLOv8 weights.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default="rescue_human_detector")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project="runs/detect",
        pretrained=True,
    )


if __name__ == "__main__":
    main()
