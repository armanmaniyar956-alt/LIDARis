"""
tests/test_adaptive_resolution.py
=================================
Unit tests for the Adaptive Variable-Resolution 2.5D Elevation Mapping Engine.
"""

import numpy as np
import pytest

from src.adaptive_resolution import (
    AdaptiveElevationGrid25D,
    ResolutionLevel,
    AdaptiveCell25D,
)


def test_adaptive_variable_sizes():
    """Verify that the adaptive grid creates cells with differing spatial dimensions."""
    # Near points (d < 3) and Far flat points (d > 14)
    near_pts = np.array([
        [1.0, 1.0, 0.5],
        [1.2, 1.2, 0.5],
    ])
    far_flat_pts = np.array([
        [14.0, 14.0, 0.0],
        [14.5, 14.5, 0.01],
    ])
    pts = np.vstack([near_pts, far_flat_pts])

    grid = AdaptiveElevationGrid25D(
        base_resolution=0.2,
        multipliers=(1, 2, 4),
        near_distance_threshold=6.0,
        far_distance_threshold=12.0
    ).fit(pts)

    assert grid.is_fitted
    sizes = set(c.cell_size for c in grid.cells)
    # Check that both fine (0.2m) and coarse (0.8m) exist
    assert 0.2 in sizes
    assert 0.8 in sizes


def test_distance_baseline_assignment():
    """Verify baseline assignment: Near -> Fine, Far -> Coarse (when flat)."""
    grid = AdaptiveElevationGrid25D(
        base_resolution=0.2,
        multipliers=(1, 2, 4),
        near_distance_threshold=5.0,
        far_distance_threshold=10.0
    )

    # 1. Point at distance 2m (Near)
    near_pts = np.array([[1.5, 1.5, 0.0]])
    grid.fit(near_pts, bounds=(-16, 16, -16, 16))

    cell_near = grid.get_cell_at(1.5, 1.5)
    assert cell_near is not None
    assert cell_near.level == ResolutionLevel.FINE
    assert "Near sensor" in cell_near.decision_reason

    # 2. Point at distance 14m (Far and Flat)
    cell_far = grid.get_cell_at(10.0, 10.0)  # distance ~ 14.1m
    assert cell_far is not None
    assert cell_far.level == ResolutionLevel.COARSE


def test_complexity_refinement():
    """Verify that a far region with a tall obstacle is refined to FINE."""
    grid = AdaptiveElevationGrid25D(
        base_resolution=0.2,
        multipliers=(1, 2, 4),
        near_distance_threshold=5.0,
        far_distance_threshold=10.0,
        height_diff_threshold=0.3
    )

    # Obstacle at distance 14m (x=10, y=10) with elevation difference 1.5m
    pts = np.array([
        [10.1, 10.1, 0.0],
        [10.2, 10.2, 1.5],
    ])
    grid.fit(pts, bounds=(0, 16, 0, 16))

    cell = grid.get_cell_at(10.1, 10.1)
    assert cell is not None
    assert cell.level == ResolutionLevel.FINE
    assert cell.is_refined is True
    assert "High complexity" in cell.decision_reason or "Obstacle detected" in cell.decision_reason


def test_temporal_dynamic_refinement():
    """Verify that moving points across frames trigger dynamic refinement."""
    grid = AdaptiveElevationGrid25D(
        base_resolution=0.2,
        multipliers=(1, 2, 4),
        near_distance_threshold=5.0,
        far_distance_threshold=10.0,
        height_diff_threshold=0.3
    )

    # Frame 0: Flat ground at (12, 12)
    t0_pts = np.array([[12.2, 12.2, 0.0]])
    # Frame 1: Moving obstacle enters at (12, 12)
    t1_pts = np.array([[12.2, 12.2, 1.2]])

    grid.fit(current_points=t1_pts, previous_points=t0_pts, bounds=(0, 16, 0, 16))
    cell = grid.get_cell_at(12.2, 12.2)

    assert cell is not None
    assert cell.level == ResolutionLevel.FINE
    assert cell.is_refined is True
    assert "DYNAMIC motion" in cell.decision_reason


def test_metrics_and_rasterization():
    """Verify summary metrics calculation and uniform rasterization."""
    pts = np.array([
        [1.0, 1.0, 1.0],
        [14.0, 14.0, 0.0],
    ])
    grid = AdaptiveElevationGrid25D(base_resolution=0.2).fit(pts, bounds=(-16, 16, -16, 16))

    metrics = grid.get_summary_metrics()
    assert metrics["total_adaptive_cells"] > 0
    assert metrics["cell_count_reduction_percentage"] > 0.0

    raster, bounds = grid.rasterize_to_uniform_matrix(resolution=0.4)
    assert raster.ndim == 2
    assert raster.shape[0] > 0 and raster.shape[1] > 0
