"""
tests/test_segmentation.py
==========================
Unit tests for geometric segmentation and segmentation interface.
"""

import numpy as np
import pytest

from src.pointcloud_io import generate_synthetic_lidar_scene
from src.segmentation_interface import (
    SemanticClass,
    GeometricGroundObstacleSegmenter,
    DeepLearningSegmentationFutureInterface,
)


def test_geometric_segmentation_ground_and_obstacles():
    pts = generate_synthetic_lidar_scene(num_ground_points=1200, seed=42)
    segmenter = GeometricGroundObstacleSegmenter()
    labels = segmenter.segment(pts)

    assert len(labels) == len(pts)
    assert set(labels).issubset({SemanticClass.UNLABELED, SemanticClass.GROUND, SemanticClass.STATIC_OBSTACLE})
    # Must have found both ground and obstacle points
    assert np.sum(labels == SemanticClass.GROUND) > 500
    assert np.sum(labels == SemanticClass.STATIC_OBSTACLE) > 50


def test_deep_learning_interface_future_error():
    dl_seg = DeepLearningSegmentationFutureInterface()
    with pytest.raises(NotImplementedError):
        dl_seg.segment(np.array([[0, 0, 0]]))
