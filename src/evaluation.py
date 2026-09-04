"""
evaluation.py
=============
Empirical Evaluation Module for comparing Fixed vs. Adaptive 2.5D LiDAR Mapping.
Problem Statement: SIH26053

Measures REAL experimental quantities directly from code execution:
- Total cells generated
- Occupied cells count
- Execution / processing runtime in milliseconds
- Workload / cell reduction percentage
- Breakdown of resolution levels (% Fine, % Medium, % Coarse)
- Approximate memory footprint
- Elevation reconstruction error (RMSE against source point elevations)
"""

import sys
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np

from src.mapping_25d import FixedElevationGrid25D
from src.adaptive_resolution import AdaptiveElevationGrid25D, ResolutionLevel


def compute_elevation_rmse(
    points: np.ndarray,
    grid_type: str,
    grid_obj: Any
) -> float:
    """
    Compute Root Mean Squared Error (RMSE) between actual point Z values
    and the grid cell mean/max elevation.

    Args:
        points: (N, 3) point cloud.
        grid_type: 'fixed' or 'adaptive'.
        grid_obj: Fitted grid instance.

    Returns:
        float: Measured elevation RMSE in meters.
    """
    if len(points) == 0:
        return 0.0

    errors = []

    if grid_type == "fixed":
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        cols = np.floor((x - grid_obj.min_x) / grid_obj.resolution).astype(np.int32)
        rows = np.floor((y - grid_obj.min_y) / grid_obj.resolution).astype(np.int32)

        valid = (
            (cols >= 0) & (cols < grid_obj.num_cells_x) &
            (rows >= 0) & (rows < grid_obj.num_cells_y)
        )
        cols = cols[valid]
        rows = rows[valid]
        z_valid = z[valid]

        cell_max_z = grid_obj.max_z[rows, cols]
        has_val = ~np.isnan(cell_max_z)
        if np.any(has_val):
            err = z_valid[has_val] - cell_max_z[has_val]
            return float(np.sqrt(np.mean(err ** 2)))
        return 0.0

    elif grid_type == "adaptive":
        for pt in points:
            cell = grid_obj.get_cell_at(pt[0], pt[1])
            if cell is not None and not np.isnan(cell.max_z):
                errors.append(pt[2] - cell.max_z)

        if len(errors) > 0:
            arr = np.array(errors, dtype=np.float64)
            return float(np.sqrt(np.mean(arr ** 2)))
        return 0.0

    return 0.0


def evaluate_fixed_vs_adaptive(
    points: np.ndarray,
    base_resolution: float = 0.2,
    multipliers: Tuple[int, int, int] = (1, 2, 4),
    bounds: Optional[Tuple[float, float, float, float]] = None,
    near_distance_threshold: float = 6.0,
    far_distance_threshold: float = 12.0,
    height_diff_threshold: float = 0.25,
    previous_points: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Run empirical benchmarking comparing Fixed vs. Adaptive 2.5D Mapping.
    All returned values are measured directly during execution.
    """
    if len(points) == 0:
        raise ValueError("Cannot evaluate empty point cloud.")

    # 1. Benchmark Baseline Fixed-Resolution Grid
    t0_fixed = time.perf_counter()
    fixed_grid = FixedElevationGrid25D(
        resolution=base_resolution,
        bounds=bounds
    ).fit(points)
    time_fixed_ms = (time.perf_counter() - t0_fixed) * 1000.0

    fixed_metrics = fixed_grid.get_metrics()
    rmse_fixed = compute_elevation_rmse(points, "fixed", fixed_grid)

    # 2. Benchmark Adaptive-Resolution Grid
    t0_adapt = time.perf_counter()
    adaptive_grid = AdaptiveElevationGrid25D(
        base_resolution=base_resolution,
        multipliers=multipliers,
        near_distance_threshold=near_distance_threshold,
        far_distance_threshold=far_distance_threshold,
        height_diff_threshold=height_diff_threshold
    ).fit(
        current_points=points,
        previous_points=previous_points,
        bounds=bounds
    )
    time_adapt_ms = (time.perf_counter() - t0_adapt) * 1000.0

    adaptive_metrics = adaptive_grid.get_summary_metrics()
    rmse_adaptive = compute_elevation_rmse(points, "adaptive", adaptive_grid)

    # Calculate actual memory footprint approximation (in KB)
    fixed_kb = fixed_metrics["approx_memory_kb"]
    # Adaptive cell list memory estimation
    adaptive_bytes = len(adaptive_grid.cells) * sys.getsizeof(adaptive_grid.cells[0]) if adaptive_grid.cells else 0
    adaptive_kb = adaptive_bytes / 1024.0

    total_fixed_cells = fixed_metrics["total_cells"]
    total_adaptive_cells = adaptive_metrics["total_adaptive_cells"]

    savings_ratio = (
        (total_fixed_cells - total_adaptive_cells) / total_fixed_cells
        if total_fixed_cells > 0 else 0.0
    )

    return {
        "fixed": {
            "resolution_m": base_resolution,
            "total_cells": total_fixed_cells,
            "occupied_cells": fixed_metrics["occupied_cells"],
            "empty_cells": fixed_metrics["empty_cells"],
            "runtime_ms": time_fixed_ms,
            "elevation_rmse_m": rmse_fixed,
            "approx_memory_kb": fixed_kb,
            "grid_obj": fixed_grid,
        },
        "adaptive": {
            "fine_resolution_m": adaptive_metrics["base_resolution_m"],
            "medium_resolution_m": adaptive_metrics["medium_resolution_m"],
            "coarse_resolution_m": adaptive_metrics["coarse_resolution_m"],
            "total_cells": total_adaptive_cells,
            "occupied_cells": adaptive_metrics["occupied_cells"],
            "empty_cells": adaptive_metrics["empty_cells"],
            "fine_cells_count": adaptive_metrics["fine_cells_count"],
            "medium_cells_count": adaptive_metrics["medium_cells_count"],
            "coarse_cells_count": adaptive_metrics["coarse_cells_count"],
            "refined_cells_count": adaptive_metrics["refined_cells_count"],
            "fine_pct": adaptive_metrics["fine_percentage"],
            "medium_pct": adaptive_metrics["medium_percentage"],
            "coarse_pct": adaptive_metrics["coarse_percentage"],
            "runtime_ms": time_adapt_ms,
            "elevation_rmse_m": rmse_adaptive,
            "approx_memory_kb": adaptive_kb,
            "grid_obj": adaptive_grid,
        },
        "comparison": {
            "cell_reduction_count": total_fixed_cells - total_adaptive_cells,
            "cell_reduction_percentage": savings_ratio * 100.0,
            "rmse_difference_m": abs(rmse_adaptive - rmse_fixed),
            "speedup_factor": (time_fixed_ms / time_adapt_ms) if time_adapt_ms > 0 else 1.0,
        }
    }
