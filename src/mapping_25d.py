"""
mapping_25d.py
==============
Baseline Fixed-Resolution 2.5D Elevation Grid Mapping Module for LIDARis.

Concepts:
- 2.5D Elevation Grid: Projects 3D point cloud onto a regular 2D (X, Y) spatial grid.
  Each grid cell stores elevation metrics:
  - max_z: Highest point in cell (critical for obstacle detection and clearance)
  - min_z: Lowest point in cell (ground/surface estimation)
  - mean_z: Average elevation
  - var_z: Elevation variance (surface roughness/obstacle complexity)
  - point_count: Density of points falling into the cell
- Empty cells: Preserved and handled cleanly using configurable values (default: np.nan).
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np


class FixedElevationGrid25D:
    """
    Fixed-resolution 2.5D Elevation Grid.
    
    Converts 3D point clouds (X, Y, Z) into a structured 2D raster grid with
    elevation and surface characteristics.
    """

    def __init__(
        self,
        resolution: float = 0.2,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        empty_value: float = np.nan
    ):
        """
        Args:
            resolution: Grid cell size in meters (fixed X and Y resolution).
            bounds: Optional (min_x, max_x, min_y, max_y). If None, calculated from points.
            empty_value: Value used to represent empty/unobserved cells. Default is np.nan.
        """
        if resolution <= 0:
            raise ValueError(f"Resolution must be positive, got {resolution}")
        self.resolution = float(resolution)
        self.bounds = bounds
        self.empty_value = empty_value

        # Grid state after fit()
        self.min_x: float = 0.0
        self.max_x: float = 0.0
        self.min_y: float = 0.0
        self.max_y: float = 0.0
        self.width_m: float = 0.0
        self.height_m: float = 0.0
        self.num_cells_x: int = 0
        self.num_cells_y: int = 0

        # Statistics matrices (shape: [num_cells_y, num_cells_x])
        self.max_z: Optional[np.ndarray] = None
        self.min_z: Optional[np.ndarray] = None
        self.mean_z: Optional[np.ndarray] = None
        self.var_z: Optional[np.ndarray] = None
        self.point_count: Optional[np.ndarray] = None
        self.occupied_mask: Optional[np.ndarray] = None

        self.is_fitted: bool = False

    def fit(self, points: np.ndarray) -> "FixedElevationGrid25D":
        """
        Populate the 2.5D grid from an (N, 3) point cloud.

        Args:
            points: (N, 3) array of X, Y, Z coordinates.

        Returns:
            self: The fitted FixedElevationGrid25D instance.
        """
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected points array of shape (N, >=3), got {points.shape}")

        if len(points) == 0:
            raise ValueError("Cannot fit grid on empty point cloud.")

        # 1. Establish spatial boundaries
        if self.bounds is not None:
            self.min_x, self.max_x, self.min_y, self.max_y = self.bounds
        else:
            self.min_x = float(np.floor(np.nanmin(points[:, 0])))
            self.max_x = float(np.ceil(np.nanmax(points[:, 0])))
            self.min_y = float(np.floor(np.nanmin(points[:, 1])))
            self.max_y = float(np.ceil(np.nanmax(points[:, 1])))

        self.width_m = max(self.max_x - self.min_x, self.resolution)
        self.height_m = max(self.max_y - self.min_y, self.resolution)

        self.num_cells_x = int(np.ceil(self.width_m / self.resolution))
        self.num_cells_y = int(np.ceil(self.height_m / self.resolution))

        # 2. Filter points strictly within spatial bounds
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        in_bounds = (
            (x >= self.min_x) & (x < self.max_x) &
            (y >= self.min_y) & (y < self.max_y) &
            (~np.isnan(z))
        )
        valid_x = x[in_bounds]
        valid_y = y[in_bounds]
        valid_z = z[in_bounds]

        # 3. Initialize grid matrices
        shape = (self.num_cells_y, self.num_cells_x)
        self.max_z = np.full(shape, self.empty_value, dtype=np.float64)
        self.min_z = np.full(shape, self.empty_value, dtype=np.float64)
        self.mean_z = np.full(shape, self.empty_value, dtype=np.float64)
        self.var_z = np.full(shape, self.empty_value, dtype=np.float64)
        self.point_count = np.zeros(shape, dtype=np.int32)
        self.occupied_mask = np.zeros(shape, dtype=bool)

        if len(valid_z) == 0:
            self.is_fitted = True
            return self

        # 4. Map coordinates to 2D grid cell indices
        col_indices = np.floor((valid_x - self.min_x) / self.resolution).astype(np.int32)
        row_indices = np.floor((valid_y - self.min_y) / self.resolution).astype(np.int32)

        # Clamp indices to ensure boundary safety
        col_indices = np.clip(col_indices, 0, self.num_cells_x - 1)
        row_indices = np.clip(row_indices, 0, self.num_cells_y - 1)

        # 5. Compute cell statistics using flat 1D linear indices for high performance
        linear_indices = row_indices * self.num_cells_x + col_indices
        total_cells = self.num_cells_y * self.num_cells_x

        # Calculate counts per cell
        counts = np.bincount(linear_indices, minlength=total_cells)
        occupied_linear = np.nonzero(counts)[0]

        if len(occupied_linear) > 0:
            # Sort points by linear cell index for group operations
            sort_order = np.argsort(linear_indices)
            sorted_linear = linear_indices[sort_order]
            sorted_z = valid_z[sort_order]

            # Find boundary splits where cell index changes
            cell_starts = np.unique(sorted_linear, return_index=True)[1]
            cell_ends = np.append(cell_starts[1:], len(sorted_z))

            # Vectorized aggregation per occupied cell
            for start, end in zip(cell_starts, cell_ends):
                cell_idx = sorted_linear[start]
                r = cell_idx // self.num_cells_x
                c = cell_idx % self.num_cells_x

                pts_z = sorted_z[start:end]
                self.max_z[r, c] = np.max(pts_z)
                self.min_z[r, c] = np.min(pts_z)
                self.mean_z[r, c] = np.mean(pts_z)
                self.var_z[r, c] = np.var(pts_z) if len(pts_z) > 1 else 0.0
                self.point_count[r, c] = len(pts_z)
                self.occupied_mask[r, c] = True

        self.is_fitted = True
        return self

    @property
    def elevation_range(self) -> np.ndarray:
        """Obstacle height clearance: max_z - min_z per cell."""
        if not self.is_fitted or self.max_z is None:
            raise RuntimeError("Grid has not been fitted yet.")
        diff = self.max_z - self.min_z
        return np.where(self.occupied_mask, diff, self.empty_value)

    def get_metrics(self) -> Dict[str, Any]:
        """Return measurable structural metrics of the fixed grid."""
        if not self.is_fitted:
            raise RuntimeError("Grid has not been fitted yet.")

        total_cells = int(self.num_cells_x * self.num_cells_y)
        occupied_cells = int(np.sum(self.occupied_mask))
        empty_cells = total_cells - occupied_cells
        occupation_ratio = occupied_cells / total_cells if total_cells > 0 else 0.0

        # Estimate memory usage of the 5 main matrices (float64: 8 bytes, int32: 4 bytes, bool: 1 byte)
        approx_memory_bytes = (
            4 * (total_cells * 8) +  # max_z, min_z, mean_z, var_z
            1 * (total_cells * 4) +  # point_count
            1 * (total_cells * 1)    # occupied_mask
        )

        return {
            "type": "FixedElevationGrid25D",
            "resolution_m": self.resolution,
            "bounds": (self.min_x, self.max_x, self.min_y, self.max_y),
            "grid_dimensions": (self.num_cells_y, self.num_cells_x),
            "total_cells": total_cells,
            "occupied_cells": occupied_cells,
            "empty_cells": empty_cells,
            "occupation_ratio": occupation_ratio,
            "approx_memory_kb": approx_memory_bytes / 1024.0,
        }
