"""Unit tests for the pinhole bearing math (spec §6)."""

import math

import pytest

from src.geo.bearing import bearing_to_target, focal_length_px, normalize_bearing_deg
from tests.synthetic import SYNTH_CAMERA, project_to_pixel, true_bearing_deg, wrap180

W = SYNTH_CAMERA.image_width
FOV = SYNTH_CAMERA.fov_horizontal_deg


def test_focal_length_known_value() -> None:
    # fov 90° -> tan(45°) = 1 -> f_px = W/2 exactly
    assert focal_length_px(1920, 90.0) == pytest.approx(960.0)


def test_center_pixel_is_heading_plus_yaw() -> None:
    assert bearing_to_target(W / 2, W, FOV, heading_deg=137.0, yaw_offset_deg=2.0) == (
        pytest.approx(139.0)
    )


def test_frame_edges_are_half_fov() -> None:
    # u = 0 -> alpha = -fov/2; u = W -> alpha = +fov/2 (exact pinhole property)
    left = bearing_to_target(0.0, W, FOV, heading_deg=0.0, yaw_offset_deg=0.0)
    right = bearing_to_target(float(W), W, FOV, heading_deg=0.0, yaw_offset_deg=0.0)
    assert left == pytest.approx(normalize_bearing_deg(-FOV / 2))
    assert right == pytest.approx(FOV / 2)


def test_bearing_wraps_past_north() -> None:
    b = bearing_to_target(W / 2, W, FOV, heading_deg=355.0, yaw_offset_deg=10.0)
    assert b == pytest.approx(5.0)
    assert 0.0 <= b < 360.0


def test_projection_roundtrip() -> None:
    """project_to_pixel (fixture) and bearing_to_target must be exact inverses."""
    camera = (37.0, 150.0)
    for vehicle_n in (0.0, 40.0, 80.0, 110.0):
        for heading in (0.0, 12.0, 350.0):
            u = project_to_pixel(camera, 5.0, vehicle_n, heading, SYNTH_CAMERA)
            if u is None:
                continue
            recovered = bearing_to_target(
                u, W, FOV, heading_deg=heading, yaw_offset_deg=SYNTH_CAMERA.yaw_offset_deg
            )
            truth = true_bearing_deg(5.0, vehicle_n, camera[0], camera[1])
            assert wrap180(recovered - truth) == pytest.approx(0.0, abs=1e-9)


def test_wrap180_range() -> None:
    for deg, expected in [(0.0, 0.0), (180.0, 180.0), (-180.0, 180.0), (200.0, -160.0)]:
        assert wrap180(deg) == pytest.approx(expected)
    assert math.isclose(wrap180(359.0), -1.0)
