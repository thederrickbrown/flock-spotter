"""Local tangent-plane (ENU) frame: lat/lon <-> meters.

Equirectangular approximation around a fixed origin. Over a single drive
(tens of km at most) the error vs a proper projection is centimeters —
negligible against GPS noise. Isolated here so pyproj can swap in later
without touching the geometry code.

Conventions: east = +x, north = +y, meters. Bearings elsewhere in the
codebase are degrees clockwise from true north.
"""

import math
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class LocalFrame:
    lat0_deg: float
    lon0_deg: float

    @classmethod
    def centered_on(cls, lats_deg: np.ndarray, lons_deg: np.ndarray) -> "LocalFrame":
        return cls(float(np.mean(lats_deg)), float(np.mean(lons_deg)))

    def to_enu(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Degrees -> (east_m, north_m). Accepts scalars or arrays."""
        cos_lat0 = math.cos(math.radians(self.lat0_deg))
        east = np.radians(np.asarray(lon_deg, dtype=float) - self.lon0_deg) * cos_lat0
        north = np.radians(np.asarray(lat_deg, dtype=float) - self.lat0_deg)
        return east * EARTH_RADIUS_M, north * EARTH_RADIUS_M

    def to_latlon(self, east_m: np.ndarray, north_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(east_m, north_m) -> degrees. Accepts scalars or arrays."""
        cos_lat0 = math.cos(math.radians(self.lat0_deg))
        lat = self.lat0_deg + np.degrees(np.asarray(north_m, dtype=float) / EARTH_RADIUS_M)
        lon = self.lon0_deg + np.degrees(
            np.asarray(east_m, dtype=float) / (EARTH_RADIUS_M * cos_lat0)
        )
        return lat, lon
