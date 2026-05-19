from datetime import datetime
from typing import Dict, List, Optional


def build_alert_message(mode: str, detections: List[Dict], gps: Optional[Dict] = None) -> str:
    """Create a concise SOS-style alert for dashboard display or future SMS/MQTT use."""
    if not detections:
        return "No victim detected."

    best = max(detections, key=lambda item: item["confidence"])
    location = ""
    if gps:
        location = f" | GPS: {gps['latitude']:.5f}, {gps['longitude']:.5f}"

    return (
        f"SOS ALERT | {mode.title()} Mode | Victims: {len(detections)} | "
        f"Best confidence: {best['confidence'] * 100:.1f}% | "
        f"Time: {datetime.now().strftime('%H:%M:%S')}{location}"
    )
