"""Pixel column -> absolute bearing (spec §6).

Pinhole model:
    f_px = (W / 2) / tan(θ_h / 2)
    α    = atan2(u - W/2, f_px)
    β    = normalize(h + yaw_offset + α)

All bearings are degrees clockwise from true north, in [0, 360).
"""

import math


def normalize_bearing_deg(bearing_deg: float) -> float:
    """Wrap any angle to [0, 360)."""
    return bearing_deg % 360.0


def focal_length_px(image_width: int, fov_horizontal_deg: float) -> float:
    """Pinhole focal length in pixels from image width and horizontal FOV."""
    return (image_width / 2.0) / math.tan(math.radians(fov_horizontal_deg) / 2.0)


def bearing_to_target(
    center_x_px: float,
    image_width: int,
    fov_horizontal_deg: float,
    heading_deg: float,
    yaw_offset_deg: float,
) -> float:
    """Absolute bearing to the object centered at pixel column `center_x_px`.

    center_x_px: horizontal center of the bounding box, pixels from left edge.
    heading_deg: vehicle heading from GPX, degrees clockwise from true north.
    yaw_offset_deg: camera mount rotation vs vehicle forward (positive = right).
    """
    f_px = focal_length_px(image_width, fov_horizontal_deg)
    alpha_deg = math.degrees(math.atan2(center_x_px - image_width / 2.0, f_px))
    return normalize_bearing_deg(heading_deg + yaw_offset_deg + alpha_deg)
