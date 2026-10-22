"""Clustering tests: connected components of the eps-graph (DBSCAN, min_samples=1)."""

import numpy as np
import pytest

from src.geo.cluster import cluster_points


def test_two_passes_collapse_to_one_candidate() -> None:
    points = np.array([[0.0, 0.0], [5.0, 3.0], [200.0, 0.0]])
    clusters = cluster_points(points, ["t1", "t2", "t3"], eps_m=15.0)
    assert [c.size for c in clusters] == [2, 1]
    assert set(clusters[0].track_ids) == {"t1", "t2"}
    np.testing.assert_allclose(clusters[0].centroid_enu, [2.5, 1.5])


def test_chaining_behavior() -> None:
    """DBSCAN chains: 0 and 24 are >eps apart but linked through 12."""
    points = np.array([[0.0, 0.0], [12.0, 0.0], [24.0, 0.0]])
    clusters = cluster_points(points, ["a", "b", "c"], eps_m=15.0)
    assert len(clusters) == 1
    assert clusters[0].size == 3


def test_empty_input() -> None:
    assert cluster_points(np.empty((0, 2)), [], eps_m=15.0) == []


def test_min_samples_other_than_one_is_refused() -> None:
    with pytest.raises(ValueError, match="min_samples"):
        cluster_points(np.array([[0.0, 0.0]]), ["a"], eps_m=15.0, min_samples=2)


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        cluster_points(np.array([[0.0, 0.0]]), ["a", "b"], eps_m=15.0)
