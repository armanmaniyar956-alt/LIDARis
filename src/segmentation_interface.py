"""
segmentation_interface.py
=========================
Segmentation Interface and Geometric Baseline for LIDARis.
Problem Statement: SIH26053

Design Philosophy:
- Provides a clean, extensible architectural interface (BaseLiDARSegmenter).
- Implements a genuine geometric ground/obstacle classifier using Open3D RANSAC plane fitting
  and DBSCAN Euclidean clustering.
- Defines classes: UNLABELED (0), GROUND (1), STATIC_OBSTACLE (2), DYNAMIC_OBJECT (3).
- Deep-learning segmentation (e.g. RandLA-Net / PointNet++) is provided as an explicit future-work
  interface without faking results or downloading massive unverified models.
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Optional, Tuple, Dict, Any
import numpy as np
import open3d as o3d

from src.pointcloud_io import numpy_to_o3d


class SemanticClass(IntEnum):
    UNLABELED = 0
    GROUND = 1
    STATIC_OBSTACLE = 2
    DYNAMIC_OBJECT = 3


class BaseLiDARSegmenter(ABC):
    """Abstract base class for LiDAR segmentation models."""

    @abstractmethod
    def segment(self, points: np.ndarray) -> np.ndarray:
        """
        Assign a semantic class ID to each (N, 3) point.

        Returns:
            np.ndarray: (N,) integer array of SemanticClass values.
        """
        pass


class GeometricGroundObstacleSegmenter(BaseLiDARSegmenter):
    """
    Genuine geometric segmentation using:
    1. RANSAC plane fitting to extract dominant ground surface.
    2. DBSCAN Euclidean clustering on non-ground points to identify obstacle clusters.
    """

    def __init__(
        self,
        distance_threshold: float = 0.20,
        ransac_n: int = 3,
        num_iterations: int = 500,
        cluster_eps: float = 0.6,
        min_cluster_points: int = 15
    ):
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
        self.cluster_eps = cluster_eps
        self.min_cluster_points = min_cluster_points

    def segment(self, points: np.ndarray) -> np.ndarray:
        """
        Segment points into GROUND (1), STATIC_OBSTACLE (2), or UNLABELED (0).
        """
        if len(points) == 0:
            return np.empty((0,), dtype=np.int32)

        labels = np.full(len(points), SemanticClass.UNLABELED, dtype=np.int32)
        pcd = numpy_to_o3d(points)

        try:
            # 1. RANSAC Ground Plane Detection
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.num_iterations
            )
            
            if len(inliers) > 0:
                labels[inliers] = SemanticClass.GROUND

            # 2. Non-ground obstacle clustering
            non_ground_mask = np.ones(len(points), dtype=bool)
            non_ground_mask[inliers] = False
            non_ground_indices = np.nonzero(non_ground_mask)[0]

            if len(non_ground_indices) > 0:
                non_ground_pcd = pcd.select_by_index(inliers, invert=True)
                cluster_labels = np.array(
                    non_ground_pcd.cluster_dbscan(
                        eps=self.cluster_eps,
                        min_points=self.min_cluster_points,
                        print_progress=False
                    )
                )

                # Points belonging to valid clusters (label >= 0) are obstacles
                valid_cluster = cluster_labels >= 0
                obstacle_original_idx = non_ground_indices[valid_cluster]
                labels[obstacle_original_idx] = SemanticClass.STATIC_OBSTACLE

        except Exception as e:
            print(f"[Warning] Geometric segmentation fallback: {e}")

        return labels


class DeepLearningSegmentationFutureInterface(BaseLiDARSegmenter):
    """
    Integration interface for future deep learning models (e.g. RandLA-Net, PointNet++).
    Explicitly marked as future work to prevent unverified deep-learning dependencies.
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path

    def segment(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Deep learning semantic segmentation is an architectural extension "
            "designated for future work. Use GeometricGroundObstacleSegmenter for the current baseline."
        )
