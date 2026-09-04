"""
tests/test_mapping_25d.py
=========================
Unit tests for the Baseline Fixed-Resolution 2.5D Elevation Grid.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.mapping_25d import FixedElevationGrid25D
from src.visualizer import plot_fixed_elevation_grid, plot_pointcloud_vs_fixed_grid


def test_grid_initialization_invalid_resolution():
    with pytest.raises(ValueError):
        FixedElevationGrid25D(resolution=-0.5)
    with pytest.raises(ValueError):
        FixedElevationGrid25D(resolution=0.0)


def test_grid_fit_and_metrics():
    # Construct synthetic points with known values
    # Cell 1: at (0.1, 0.1), two points with z=1.0 and z=3.0
    # Cell 2: at (0.9, 0.9), one point with z=5.0
    pts = np.array([
        [0.1, 0.1, 1.0],
        [0.1, 0.1, 3.0],
        [0.9, 0.9, 5.0],
    ])

    grid = FixedElevationGrid25D(resolution=0.5, bounds=(0.0, 1.0, 0.0, 1.0))
    grid.fit(pts)

    assert grid.is_fitted
    assert grid.num_cells_x == 2
    assert grid.num_cells_y == 2
    assert grid.occupied_mask[0, 0] is np.True_ or grid.occupied_mask[0, 0] == True
    assert grid.occupied_mask[1, 1] is np.True_ or grid.occupied_mask[1, 1] == True
    assert grid.occupied_mask[0, 1] == False  # Unoccupied
    assert grid.occupied_mask[1, 0] == False  # Unoccupied

    # Check cell (0, 0): max=3.0, min=1.0, mean=2.0, var=1.0, count=2
    assert np.isclose(grid.max_z[0, 0], 3.0)
    assert np.isclose(grid.min_z[0, 0], 1.0)
    assert np.isclose(grid.mean_z[0, 0], 2.0)
    assert np.isclose(grid.var_z[0, 0], 1.0)
    assert grid.point_count[0, 0] == 2

    # Check empty cell returns NaN
    assert np.isnan(grid.max_z[0, 1])

    # Check metrics dictionary
    metrics = grid.get_metrics()
    assert metrics["total_cells"] == 4
    assert metrics["occupied_cells"] == 2
    assert metrics["empty_cells"] == 2
    assert metrics["occupation_ratio"] == 0.5


def test_elevation_range():
    pts = np.array([
        [2.0, 2.0, 0.5],
        [2.0, 2.0, 3.5],
    ])
    grid = FixedElevationGrid25D(resolution=1.0, bounds=(0.0, 5.0, 0.0, 5.0))
    grid.fit(pts)

    er = grid.elevation_range
    assert np.isclose(er[2, 2], 3.0)  # 3.5 - 0.5 = 3.0
    assert np.isnan(er[0, 0])


def test_plot_fixed_elevation_grid():
    with tempfile.TemporaryDirectory() as tmpdir:
        pts = np.random.uniform(-5, 5, size=(500, 3))
        grid = FixedElevationGrid25D(resolution=0.5).fit(pts)

        out_path = Path(tmpdir) / "test_fixed.png"
        saved = plot_fixed_elevation_grid(grid, value_key="max_z", output_path=out_path)
        assert Path(saved).exists()
        assert Path(saved).stat().st_size > 0


def test_plot_pointcloud_vs_fixed_grid():
    with tempfile.TemporaryDirectory() as tmpdir:
        pts = np.random.uniform(-10, 10, size=(1000, 3))
        grid = FixedElevationGrid25D(resolution=0.5).fit(pts)

        out_path = Path(tmpdir) / "test_compare.png"
        saved = plot_pointcloud_vs_fixed_grid(pts, grid, output_path=out_path)
        assert Path(saved).exists()
        assert Path(saved).stat().st_size > 0
