"""
tests/test_evaluation.py
========================
Unit tests for the empirical evaluation module and master comparison visualizer.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.pointcloud_io import generate_synthetic_lidar_scene
from src.evaluation import evaluate_fixed_vs_adaptive, compute_elevation_rmse
from src.visualizer import (
    plot_adaptive_grid_patches,
    plot_resolution_allocation_map,
    plot_full_sih_comparison,
)


def test_evaluate_fixed_vs_adaptive():
    pts = generate_synthetic_lidar_scene(num_ground_points=1500, seed=42)
    results = evaluate_fixed_vs_adaptive(
        pts,
        base_resolution=0.25,
        near_distance_threshold=6.0,
        far_distance_threshold=12.0
    )

    assert "fixed" in results
    assert "adaptive" in results
    assert "comparison" in results

    f = results["fixed"]
    a = results["adaptive"]
    c = results["comparison"]

    assert f["total_cells"] > 0
    assert a["total_cells"] > 0
    assert a["total_cells"] < f["total_cells"]  # True reduction
    assert c["cell_reduction_percentage"] > 0.0
    assert f["runtime_ms"] >= 0.0
    assert a["runtime_ms"] >= 0.0
    assert f["elevation_rmse_m"] >= 0.0
    assert a["elevation_rmse_m"] >= 0.0


def test_visualizers_file_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        pts = generate_synthetic_lidar_scene(num_ground_points=1000, seed=7)
        results = evaluate_fixed_vs_adaptive(pts, base_resolution=0.5)

        adapt_grid = results["adaptive"]["grid_obj"]
        fixed_grid = results["fixed"]["grid_obj"]

        # 1. Test adaptive grid patches
        p1 = tmp_path / "adaptive_patches.png"
        saved1 = plot_adaptive_grid_patches(adapt_grid, output_path=p1)
        assert Path(saved1).exists()
        assert Path(saved1).stat().st_size > 0

        # 2. Test resolution allocation map
        p2 = tmp_path / "resolution_map.png"
        saved2 = plot_resolution_allocation_map(adapt_grid, output_path=p2)
        assert Path(saved2).exists()
        assert Path(saved2).stat().st_size > 0

        # 3. Test master SIH comparison plot
        p3 = tmp_path / "full_comparison.png"
        saved3 = plot_full_sih_comparison(pts, fixed_grid, adapt_grid, results, output_path=p3)
        assert Path(saved3).exists()
        assert Path(saved3).stat().st_size > 0
