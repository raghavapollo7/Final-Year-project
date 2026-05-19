# AI-Powered Search and Rescue Drone System

Prototype computer vision system for detecting humans in two disaster rescue scenarios:

- Underwater mode: low visibility, blue-green color cast, blur, poor lighting.
- Rubble mode: partial occlusion, debris, simulated thermal/infrared view.

The demo uses Python, OpenCV, YOLOv8, PyTorch, and Streamlit. It is designed for a final year project presentation where a clean live dashboard matters as much as model accuracy.

## 1. Folder Structure

```text
ai-search-rescue-drone/
  app.py
  requirements.txt
  README.md
  config/
    app_config.yaml
    dataset_underwater.yaml
    dataset_rubble.yaml
  src/
    __init__.py
    alerts.py
    detector.py
    gps.py
    heatmap.py
    preprocessing.py
    utils.py
  scripts/
    train_yolo.py
    run_inference.py
  data/
    raw/
    processed/
  models/
    README.md
  outputs/
```

## 2. Installation

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

YOLOv8 will automatically download `yolov8n.pt` the first time you run the app. For a better demo, place your trained model in:

```text
models/best.pt
```

## 3. Run the Dashboard

```powershell
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## 4. Dataset Suggestions

Underwater human detection:

- Use underwater diver/swimmer videos from public sources and label humans manually with Roboflow, CVAT, or LabelImg.
- Search terms: "underwater diver detection dataset", "underwater swimmer dataset", "UFO-120 underwater object dataset", "SUIM underwater image dataset".
- If a dataset does not contain a `person` class, create a small custom dataset from underwater rescue/diver footage.

Rubble/disaster detection:

- Use COCO person images and augment them with occlusion patches, dust, blur, and debris overlays.
- Use disaster/search-rescue imagery from open emergency response datasets when licensing allows.
- For thermal support, use FLIR ADAS or other thermal human datasets, then map the class to `person`.

Recommended annotation format: YOLO format.

```text
dataset/
  images/train
  images/val
  labels/train
  labels/val
```

Each label file:

```text
class_id x_center y_center width height
```

Coordinates are normalized from 0 to 1.

## 5. Training

Edit `config/dataset_underwater.yaml` or `config/dataset_rubble.yaml` so the paths point to your dataset.

Train underwater model:

```powershell
python scripts/train_yolo.py --data config/dataset_underwater.yaml --weights yolov8n.pt --epochs 50 --imgsz 640 --name underwater_human_detector
```

Train rubble model:

```powershell
python scripts/train_yolo.py --data config/dataset_rubble.yaml --weights yolov8n.pt --epochs 50 --imgsz 640 --name rubble_human_detector
```

After training, copy the best checkpoint:

```powershell
copy runs\detect\underwater_human_detector\weights\best.pt models\underwater_best.pt
copy runs\detect\rubble_human_detector\weights\best.pt models\rubble_best.pt
```

## 6. Inference Without UI

```powershell
python scripts/run_inference.py --source 0 --mode underwater
python scripts/run_inference.py --source path\to\video.mp4 --mode rubble
```

## 7. Why YOLOv8?

YOLOv8 is a strong fit for this prototype because it is fast enough for webcam/video inference, has a simple training API, supports transfer learning, and produces bounding boxes with confidence scores. For a drone-style rescue demo, low latency matters: the system should show detections live rather than only after offline processing.

## 8. Main Challenges

Underwater detection is difficult because water absorbs red light, creates blue-green color distortion, reduces contrast, adds haze, and introduces blur or floating particles. The prototype handles this with white balance, CLAHE, contrast enhancement, sharpening, and optional dehazing.

Rubble detection is difficult because victims may be partially visible, covered by debris, badly lit, or viewed from unusual angles. The prototype helps by using confidence thresholding, simulated thermal visualization, heatmap accumulation, and a clean alert workflow.

## 9. Future Improvements

- Train separate underwater and rubble models on real rescue imagery.
- Add segmentation with YOLOv8-seg or SAM for victim outlines.
- Add pose estimation for partially visible limbs.
- Use real thermal/infrared camera input.
- Fuse RGB, thermal, GPS, IMU, and altitude data.
- Deploy on an edge device such as NVIDIA Jetson.
- Send SOS alerts to a control room through SMS, MQTT, or a cloud dashboard.

## 10. Professor Demo Flow

1. Start Streamlit dashboard.
2. Select Underwater Mode and show preprocessing toggle.
3. Play underwater/diver video or webcam sample.
4. Show bounding boxes, confidence, victim count, alert banner, and heatmap.
5. Switch to Rubble Mode.
6. Enable thermal simulation and show partially occluded person examples.
7. Explain how the trained model can be deployed on a drone camera feed.
