"""
tests/test_pointcloud_io.py
===========================
Unit tests for point cloud loading, saving, conversion, and visualization functions.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.pointcloud_io import (
    load_point_cloud,
    save_point_cloud,
    numpy_to_o3d,
    o3d_to_numpy,
    generate_synthetic_lidar_scene,
    get_or_create_fallback_pointcloud,
)
from src.visualizer import colorize_by_elevation, render_point_cloud_topdown


def test_synthetic_pointcloud_generation():
    points = generate_synthetic_lidar_scene(num_ground_points=1000, seed=123)
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert points.shape[0] > 1000
    assert not np.isnan(points).any()


def test_numpy_o3d_conversion():
    original = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    pcd = numpy_to_o3d(original)
    recovered = o3d_to_numpy(pcd)
    assert np.allclose(original, recovered)


def test_save_and_load_ply():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.ply"
        original = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-5.0, 3.0, 1.5],
        ], dtype=np.float64)

        success = save_point_cloud(file_path, original)
        assert success
        assert file_path.exists()

        loaded = load_point_cloud(file_path)
        assert loaded.shape == original.shape
        assert np.allclose(original, loaded, atol=1e-4)


def test_save_and_load_pcd():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.pcd"
        original = np.array([
            [1.5, 2.5, 3.5],
            [-1.0, -2.0, -3.0],
        ], dtype=np.float64)

        success = save_point_cloud(file_path, original)
        assert success
        assert file_path.exists()

        loaded = load_point_cloud(file_path)
        assert loaded.shape == original.shape
        assert np.allclose(original, loaded, atol=1e-4)


def test_save_and_load_bin():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.bin"
        original = np.array([
            [10.0, 20.0, 1.0],
            [30.0, 40.0, 2.0],
        ], dtype=np.float64)

        success = save_point_cloud(file_path, original)
        assert success
        assert file_path.exists()

        loaded = load_point_cloud(file_path)
        assert loaded.shape == original.shape
        assert np.allclose(original, loaded, atol=1e-5)


def test_save_and_load_npy():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.npy"
        original = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        success = save_point_cloud(file_path, original)
        assert success
        assert file_path.exists()

        loaded = load_point_cloud(file_path)
        assert np.allclose(original, loaded)


def test_colorize_by_elevation():
    points = np.array([[0, 0, 0.0], [0, 0, 10.0]])
    colors = colorize_by_elevation(points)
    assert colors.shape == (2, 3)
    assert (colors >= 0.0).all() and (colors <= 1.0).all()


def test_render_point_cloud_topdown():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_img = Path(tmpdir) / "topdown.png"
        points = generate_synthetic_lidar_scene(num_ground_points=500, seed=99)
        path = render_point_cloud_topdown(points, output_path=output_img)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0
