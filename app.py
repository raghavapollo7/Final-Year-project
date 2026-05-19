import time

import cv2
import numpy as np
import streamlit as st

from src.alerts import build_alert_message
from src.detector import RescueDetector
from src.gps import get_demo_gps
from src.heatmap import DetectionHeatmap
from src.preprocessing import preprocess_rubble, preprocess_underwater, simulate_thermal
from src.utils import bgr_to_rgb, ensure_output_dirs, load_yaml, open_video_source


st.set_page_config(page_title="AI Search and Rescue Drone", layout="wide")


@st.cache_resource
def load_detector(weights: str, confidence: float, iou: float):
    return RescueDetector(weights=weights, confidence=confidence, iou=iou)


def process_frame(frame, mode, preprocess_enabled, thermal_enabled, detector, heatmap, heatmap_enabled):
    if mode == "underwater":
        detection_frame = preprocess_underwater(frame) if preprocess_enabled else frame
        display_frame = detection_frame
    else:
        detection_frame = preprocess_rubble(frame, thermal=False) if preprocess_enabled else frame
        display_frame = simulate_thermal(detection_frame) if thermal_enabled else detection_frame

    _, detections = detector.detect(detection_frame)
    annotated = display_frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        RescueDetector._draw_detection(annotated, x1, y1, x2, y2, detection["confidence"])

    if heatmap_enabled:
        heat = heatmap.update(annotated.shape, detections)
        annotated = heatmap.overlay(annotated, heat)
    return annotated, detections


def main():
    ensure_output_dirs()
    config = load_yaml("config/app_config.yaml")

    st.title("AI-Powered Search and Rescue Drone System")
    st.caption("Underwater and rubble human detection with YOLOv8, OpenCV, and rescue alerts.")

    with st.sidebar:
        st.header("Mission Control")
        mode = st.radio("Detection mode", ["underwater", "rubble"], format_func=str.title)
        source = st.text_input("Camera/video source", value="0")
        uploaded_image = st.file_uploader("Demo image", type=["jpg", "jpeg", "png"])
        default_confidence = 0.20 if mode == "underwater" else float(config["model"]["confidence"])
        default_alert = 0.25 if mode == "underwater" else float(config["dashboard"]["alert_confidence"])
        confidence = st.slider("Detection confidence", 0.05, 0.90, default_confidence, 0.05)
        alert_confidence = st.slider("Alert threshold", 0.05, 0.95, default_alert, 0.05)
        preprocess_enabled = st.toggle("Preprocess video", value=True)
        thermal_enabled = st.toggle("Thermal simulation", value=(mode == "rubble"), disabled=(mode != "rubble"))
        heatmap_enabled = st.toggle("Heatmap overlay", value=True)
        st.divider()
        latitude = st.number_input("Demo latitude", value=float(config["gps"]["default_latitude"]), format="%.6f")
        longitude = st.number_input("Demo longitude", value=float(config["gps"]["default_longitude"]), format="%.6f")
        start = st.button("Start Mission", type="primary", use_container_width=True)
        stop = st.button("Stop", use_container_width=True)

    fallback_weights = config["model"]["default_weights"]
    primary_weights = (
        config["model"]["underwater_weights"] if mode == "underwater" else config["model"]["rubble_weights"]
    )
    weights = RescueDetector.choose_weights(primary_weights, fallback_weights)
    detector = load_detector(weights, confidence, float(config["model"]["iou"]))
    heatmap = DetectionHeatmap(decay=float(config["dashboard"]["heatmap_decay"]))
    gps = get_demo_gps(latitude, longitude)

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    frame_view, side_panel = st.columns([2.2, 1])
    video_slot = frame_view.empty()
    alert_slot = frame_view.empty()
    side_panel.subheader("Mission Telemetry")
    telemetry_slot = side_panel.empty()
    detection_table_slot = side_panel.empty()

    st.info(f"Active weights: {weights}")
    if mode == "underwater":
        st.warning(
            "Underwater diver detection is difficult with a generic COCO model. "
            "For demo, keep detection confidence near 0.15-0.25 and try preprocessing both ON and OFF."
        )

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Could not read uploaded image.")
            return

        annotated, detections = process_frame(
            frame, mode, preprocess_enabled, thermal_enabled, detector, heatmap, heatmap_enabled
        )
        high_conf_detections = [item for item in detections if item["confidence"] >= alert_confidence]
        metric_a.metric("Mode", mode.title())
        metric_b.metric("Victims now", len(detections))
        metric_c.metric("Alert-level", len(high_conf_detections))
        metric_d.metric("Input", "Image")

        if high_conf_detections:
            alert_slot.error(build_alert_message(mode, high_conf_detections, gps))
        elif detections:
            alert_slot.warning(
                f"Potential victim detected: {len(detections)} candidate(s), "
                f"best confidence {max(item['confidence'] for item in detections) * 100:.1f}%."
            )
        else:
            alert_slot.success("No victim candidate detected in this image.")

        telemetry_slot.json(
            {
                "mode": mode,
                "source": uploaded_image.name,
                "gps": gps,
                "preprocessing": preprocess_enabled,
                "thermal_simulation": thermal_enabled if mode == "rubble" else False,
                "all_detections": len(detections),
            }
        )
        detection_table_slot.dataframe(detections, use_container_width=True)
        video_slot.image(bgr_to_rgb(annotated), channels="RGB", use_column_width=True)
        return

    if stop:
        st.session_state["running"] = False
    if start:
        st.session_state["running"] = True

    if not st.session_state.get("running", False):
        st.warning("Select a mode and press Start Mission to begin live detection.")
        return

    cap = open_video_source(source)
    if not cap.isOpened():
        st.error("Could not open camera or video source. Try source 0, 1, or a valid video path.")
        st.session_state["running"] = False
        return

    total_victims = 0
    frame_count = 0
    started_at = time.time()

    while st.session_state.get("running", False):
        ok, frame = cap.read()
        if not ok:
            st.warning("Video stream ended or camera frame could not be read.")
            break

        frame_count += 1
        annotated, detections = process_frame(
            frame, mode, preprocess_enabled, thermal_enabled, detector, heatmap, heatmap_enabled
        )
        high_conf_detections = [item for item in detections if item["confidence"] >= alert_confidence]
        total_victims = max(total_victims, len(detections))

        fps = frame_count / max(time.time() - started_at, 1e-6)
        metric_a.metric("Mode", mode.title())
        metric_b.metric("Victims now", len(detections))
        metric_c.metric("Max count", total_victims)
        metric_d.metric("FPS", f"{fps:.1f}")

        if high_conf_detections:
            alert_slot.error(build_alert_message(mode, high_conf_detections, gps))
        elif detections:
            alert_slot.warning(
                f"Potential victim detected: {len(detections)} candidate(s), "
                f"best confidence {max(item['confidence'] for item in detections) * 100:.1f}%."
            )
        else:
            alert_slot.success("Scanning area. No victim candidate detected.")

        telemetry_slot.json(
            {
                "mode": mode,
                "source": source,
                "gps": gps,
                "preprocessing": preprocess_enabled,
                "thermal_simulation": thermal_enabled if mode == "rubble" else False,
                "all_detections": len(detections),
            }
        )
        detection_table_slot.dataframe(detections, use_container_width=True)
        video_slot.image(bgr_to_rgb(annotated), channels="RGB", use_column_width=True)

    cap.release()
    st.session_state["running"] = False


if __name__ == "__main__":
    main()
