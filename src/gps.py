from typing import Dict


def get_demo_gps(latitude: float, longitude: float) -> Dict[str, float]:
    """Return manually supplied demo GPS coordinates."""
    return {"latitude": float(latitude), "longitude": float(longitude)}
