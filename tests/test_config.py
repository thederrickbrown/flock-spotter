"""Config loading: the shipped YAML files must round-trip into typed configs."""

from pathlib import Path

import pytest

from src.config import PipelineConfig, load_cameras, load_pipeline, load_regions

REPO_ROOT = Path(__file__).parent.parent


def test_shipped_pipeline_yaml_matches_defaults() -> None:
    """config/pipeline.yaml documents the defaults — they must not drift apart."""
    assert load_pipeline(REPO_ROOT / "config" / "pipeline.yaml") == PipelineConfig()


def test_pipeline_values_are_numeric_even_from_yaml_strings(tmp_path: Path) -> None:
    """YAML 1.1 parses '1.0e4' as a STRING; the loader must coerce it.

    Regression test: a string threshold silently breaks every '>' comparison.
    """
    cfg_path = tmp_path / "p.yaml"
    cfg_path.write_text("max_condition_number: 1.0e4\nmin_rays: '4'\n")
    cfg = load_pipeline(cfg_path)
    assert cfg.max_condition_number == 1.0e4
    assert cfg.min_rays == 4
    assert isinstance(cfg.max_condition_number, float)
    assert isinstance(cfg.min_rays, int)


def test_unknown_pipeline_key_is_an_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "p.yaml"
    cfg_path.write_text("max_residualm: 9.0\n")  # typo'd key must not pass silently
    with pytest.raises(ValueError, match="unknown pipeline settings"):
        load_pipeline(cfg_path)


def test_shipped_cameras_and_regions_load() -> None:
    cameras = load_cameras(REPO_ROOT / "config" / "cameras.yaml")
    assert "iphone15-wide-mount-a" in cameras
    cam = cameras["iphone15-wide-mount-a"]
    assert cam.image_width == 1920
    assert 0.0 < cam.fov_horizontal_deg < 180.0

    regions = load_regions(REPO_ROOT / "config" / "regions.yaml")
    assert regions["san-diego"].south < regions["san-diego"].north
    assert regions["san-diego"].west < regions["san-diego"].east
