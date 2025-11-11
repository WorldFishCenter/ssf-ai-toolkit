from __future__ import annotations

import numpy as np


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Calculate great circle distance in kilometers."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Compute bearing (degrees) from (lat1, lon1) to (lat2, lon2)."""
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    x = np.sin(dLon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dLon)
    b = np.degrees(np.arctan2(x, y))
    return (b + 360) % 360
