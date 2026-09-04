"""
adaptive_resolution.py
======================
Core Adaptive Variable-Resolution 2.5D LiDAR Mapping Engine for LIDARis.
Problem Statement: SIH26053

Key Principles:
1. True Variable-Resolution Spatial Representation:
   The environment is represented by actual variable-sized spatial cells (FINE, MEDIUM, COARSE),
   not merely cosmetic colors on a fixed grid.
2. Distance-Based Baseline:
   Near regions receive fine detail; far regions receive coarse detail where sensor divergence is high.
3. Complexity and Dynamic Refinement:
   Regions with significant elevation variance (obstacles, structures), high density, or
   frame-to-frame temporal changes are dynamically refined to finer resolutions, even at large distances.
4. Explainable AI / Decision Logic:
   Every cell retains a human-readable explanation justifying why its resolution was chosen.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np


class ResolutionLevel(Enum):
    FINE = "FINE"
    MEDIUM = "MEDIUM"
    COARSE = "COARSE"


@dataclass
class AdaptiveCell25D:
    """
    Individual spatial cell in the adaptive 2.5D elevation map.
    """
    # Spatial bounds
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    center_x: float
    center_y: float
    cell_size: float

    # Resolution classification
    level: ResolutionLevel

    # Occupancy & point content
    is_occupied: bool
    point_count: int

    # 2.5D Elevation properties
    max_z: float
    min_z: float
    mean_z: float
    var_z: float
    elevation_range: float  # max_z - min_z (obstacle height clearance)

    # Decision factors & explainability
    distance_from_sensor: float
    is_refined: bool
    decision_reason: str


class AdaptiveElevationGrid25D:
    """
    Adaptive Variable-Resolution 2.5D Elevation Mapping Engine.
    
    Dynamically assigns spatial resolution (FINE, MEDIUM, COARSE) across the scene
    based on sensor distance and local surface complexity (variance/obstacles).
    """

    def __init__(
        self,
        base_resolution: float = 0.2,
        multipliers: Tuple[int, int, int] = (1, 2, 4),
        near_distance_threshold: float = 6.0,
        far_distance_threshold: float = 12.0,
        height_diff_threshold: float = 0.25,
        variance_threshold: float = 0.04,
        sensor_origin: Tuple[float, float] = (0.0, 0.0),
        empty_value: float = np.nan
    ):
        """
        Args:
            base_resolution: Size of finest cell in meters (e.g., 0.2m).
            multipliers: (fine_mult, medium_mult, coarse_mult), typically (1, 2, 4).
                         Yields: Fine=0.2m, Medium=0.4m, Coarse=0.8m.
            near_distance_threshold: Distances <= this use FINE baseline.
            far_distance_threshold: Distances > this use COARSE baseline. Distances in-between use MEDIUM.
            height_diff_threshold: Elevation range (max_z - min_z) in meters triggering complexity refinement.
            variance_threshold: Elevation variance in m² triggering complexity refinement.
            sensor_origin: (X, Y) coordinate of LiDAR sensor.
            empty_value: Value for unobserved elevations.
        """
        self.base_resolution = float(base_resolution)
        self.fine_size = self.base_resolution * multipliers[0]
        self.medium_size = self.base_resolution * multipliers[1]
        self.coarse_size = self.base_resolution * multipliers[2]

        self.near_distance_threshold = float(near_distance_threshold)
        self.far_distance_threshold = float(far_distance_threshold)
        self.height_diff_threshold = float(height_diff_threshold)
        self.variance_threshold = float(variance_threshold)
        self.sensor_origin = sensor_origin
        self.empty_value = empty_value

        # State storage
        self.cells: List[AdaptiveCell25D] = []
        self.min_x: float = 0.0
        self.max_x: float = 0.0
        self.min_y: float = 0.0
        self.max_y: float = 0.0
        self.is_fitted: bool = False

    def _determine_resolution(
        self,
        distance: float,
        pts_in_block: np.ndarray,
        prev_block_pts: Optional[np.ndarray] = None
    ) -> Tuple[ResolutionLevel, bool, str]:
        """
        Determine target resolution and generate an explainable justification.
        """
        # Step 1: Distance-based baseline
        if distance <= self.near_distance_threshold:
            baseline = ResolutionLevel.FINE
            base_reason = f"Near sensor (range {distance:.1f}m <= {self.near_distance_threshold:.1f}m)"
        elif distance <= self.far_distance_threshold:
            baseline = ResolutionLevel.MEDIUM
            base_reason = f"Mid sensor (range {distance:.1f}m in [{self.near_distance_threshold:.1f}m, {self.far_distance_threshold:.1f}m])"
        else:
            baseline = ResolutionLevel.COARSE
            base_reason = f"Far sensor (range {distance:.1f}m > {self.far_distance_threshold:.1f}m)"

        if len(pts_in_block) == 0:
            return ResolutionLevel.COARSE, False, f"{base_reason}; empty cell -> COARSE"

        # Step 2: Complexity & obstacle analysis
        z_vals = pts_in_block[:, 2]
        h_diff = float(np.ptp(z_vals))
        var_z = float(np.var(z_vals)) if len(z_vals) > 1 else 0.0

        is_complex = (h_diff >= self.height_diff_threshold) or (var_z >= self.variance_threshold)

        # Step 3: Dynamic / Temporal motion check
        is_dynamic = False
        temporal_shift = 0.0
        if prev_block_pts is not None and len(prev_block_pts) > 0:
            curr_mean_z = np.mean(z_vals)
            prev_mean_z = np.mean(prev_block_pts[:, 2])
            temporal_shift = abs(curr_mean_z - prev_mean_z)
            if temporal_shift > self.height_diff_threshold:
                is_dynamic = True

        # Step 4: Refinement decisions
        if is_dynamic:
            return (
                ResolutionLevel.FINE,
                True,
                f"{base_reason} + DYNAMIC motion detected (Δz_t={temporal_shift:.2f}m) -> Refined to FINE"
            )

        if baseline == ResolutionLevel.COARSE:
            if is_complex:
                # Far obstacle refined to FINE for safety-critical obstacle perception
                return (
                    ResolutionLevel.FINE,
                    True,
                    f"{base_reason} + High complexity (Δz={h_diff:.2f}m, var={var_z:.3f}m²) -> Refined to FINE"
                )
            return (
                ResolutionLevel.COARSE,
                False,
                f"{base_reason} + Low variation (Δz={h_diff:.2f}m) -> Retained COARSE"
            )

        elif baseline == ResolutionLevel.MEDIUM:
            if is_complex:
                return (
                    ResolutionLevel.FINE,
                    True,
                    f"{base_reason} + Obstacle detected (Δz={h_diff:.2f}m) -> Refined to FINE"
                )
            return (
                ResolutionLevel.MEDIUM,
                False,
                f"{base_reason} + Moderate variation -> Retained MEDIUM"
            )

        else:  # Baseline was already FINE
            return (
                ResolutionLevel.FINE,
                False,
                f"{base_reason} -> Baseline FINE"
            )

    def fit(
        self,
        current_points: np.ndarray,
        previous_points: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> "AdaptiveElevationGrid25D":
        """
        Build the adaptive variable-resolution 2.5D grid.

        Args:
            current_points: (N, 3) current LiDAR point cloud.
            previous_points: Optional (M, 3) previous frame point cloud for temporal change detection.
            bounds: Optional (min_x, max_x, min_y, max_y).
        """
        if current_points.ndim != 2 or current_points.shape[1] < 3:
            raise ValueError(f"Expected points array of shape (N, >=3), got {current_points.shape}")

        if bounds is not None:
            self.min_x, self.max_x, self.min_y, self.max_y = bounds
        else:
            self.min_x = float(np.floor(np.nanmin(current_points[:, 0])))
            self.max_x = float(np.ceil(np.nanmax(current_points[:, 0])))
            self.min_y = float(np.floor(np.nanmin(current_points[:, 1])))
            self.max_y = float(np.ceil(np.nanmax(current_points[:, 1])))

        # Align boundaries to coarse block multiples
        block_size = self.coarse_size
        self.min_x = np.floor(self.min_x / block_size) * block_size
        self.max_x = np.ceil(self.max_x / block_size) * block_size
        self.min_y = np.floor(self.min_y / block_size) * block_size
        self.max_y = np.ceil(self.max_y / block_size) * block_size

        x_blocks = int(np.round((self.max_x - self.min_x) / block_size))
        y_blocks = int(np.round((self.max_y - self.min_y) / block_size))

        self.cells = []

        # Filter valid points
        c_pts = current_points[
            (current_points[:, 0] >= self.min_x) & (current_points[:, 0] < self.max_x) &
            (current_points[:, 1] >= self.min_y) & (current_points[:, 1] < self.max_y)
        ]
        
        p_pts = None
        if previous_points is not None and len(previous_points) > 0:
            p_pts = previous_points[
                (previous_points[:, 0] >= self.min_x) & (previous_points[:, 0] < self.max_x) &
                (previous_points[:, 1] >= self.min_y) & (previous_points[:, 1] < self.max_y)
            ]

        # Fast spatial index for points into root blocks
        col_blocks = np.floor((c_pts[:, 0] - self.min_x) / block_size).astype(np.int32)
        row_blocks = np.floor((c_pts[:, 1] - self.min_y) / block_size).astype(np.int32)
        block_indices = row_blocks * x_blocks + col_blocks

        # Iterate over all macro blocks in the environment
        for by in range(y_blocks):
            for bx in range(x_blocks):
                bx_min = self.min_x + bx * block_size
                bx_max = bx_min + block_size
                by_min = self.min_y + by * block_size
                by_max = by_min + block_size
                
                cx = 0.5 * (bx_min + bx_max)
                cy = 0.5 * (by_min + by_max)
                dist = float(np.hypot(cx - self.sensor_origin[0], cy - self.sensor_origin[1]))

                current_block_idx = by * x_blocks + bx
                mask_curr = (block_indices == current_block_idx)
                pts_in_macro = c_pts[mask_curr]

                prev_pts_in_macro = None
                if p_pts is not None:
                    prev_pts_in_macro = p_pts[
                        (p_pts[:, 0] >= bx_min) & (p_pts[:, 0] < bx_max) &
                        (p_pts[:, 1] >= by_min) & (p_pts[:, 1] < by_max)
                    ]

                target_level, is_refined, reason = self._determine_resolution(
                    dist, pts_in_macro, prev_pts_in_macro
                )

                # Subdivide macro block according to chosen resolution level
                if target_level == ResolutionLevel.COARSE:
                    sub_cell_size = self.coarse_size
                    splits = 1
                elif target_level == ResolutionLevel.MEDIUM:
                    sub_cell_size = self.medium_size
                    splits = 2
                else:  # FINE
                    sub_cell_size = self.fine_size
                    splits = 4

                sub_step = block_size / splits

                for sy in range(splits):
                    for sx in range(splits):
                        cell_x_min = bx_min + sx * sub_step
                        cell_x_max = cell_x_min + sub_step
                        cell_y_min = by_min + sy * sub_step
                        cell_y_max = cell_y_min + sub_step
                        
                        center_x = 0.5 * (cell_x_min + cell_x_max)
                        center_y = 0.5 * (cell_y_min + cell_y_max)
                        cell_dist = float(np.hypot(center_x - self.sensor_origin[0], center_y - self.sensor_origin[1]))

                        # Filter points for this sub-cell
                        if len(pts_in_macro) > 0:
                            sub_mask = (
                                (pts_in_macro[:, 0] >= cell_x_min) & (pts_in_macro[:, 0] < cell_x_max) &
                                (pts_in_macro[:, 1] >= cell_y_min) & (pts_in_macro[:, 1] < cell_y_max)
                            )
                            sub_pts = pts_in_macro[sub_mask]
                        else:
                            sub_pts = np.empty((0, 3))

                        count = len(sub_pts)
                        is_occ = count > 0

                        if is_occ:
                            z_vals = sub_pts[:, 2]
                            max_z = float(np.max(z_vals))
                            min_z = float(np.min(z_vals))
                            mean_z = float(np.mean(z_vals))
                            var_z = float(np.var(z_vals)) if count > 1 else 0.0
                            elev_range = max_z - min_z
                        else:
                            max_z = self.empty_value
                            min_z = self.empty_value
                            mean_z = self.empty_value
                            var_z = self.empty_value
                            elev_range = self.empty_value

                        cell = AdaptiveCell25D(
                            x_min=cell_x_min,
                            x_max=cell_x_max,
                            y_min=cell_y_min,
                            y_max=cell_y_max,
                            center_x=center_x,
                            center_y=center_y,
                            cell_size=sub_cell_size,
                            level=target_level,
                            is_occupied=is_occ,
                            point_count=count,
                            max_z=max_z,
                            min_z=min_z,
                            mean_z=mean_z,
                            var_z=var_z,
                            elevation_range=elev_range,
                            distance_from_sensor=cell_dist,
                            is_refined=is_refined,
                            decision_reason=reason
                        )
                        self.cells.append(cell)

        self.is_fitted = True
        return self

    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Return measurable experimental characteristics of the adaptive 2.5D representation.
        """
        if not self.is_fitted:
            raise RuntimeError("Adaptive grid has not been fitted yet.")

        total_cells = len(self.cells)
        occupied_cells = sum(1 for c in self.cells if c.is_occupied)
        empty_cells = total_cells - occupied_cells

        fine_cells = sum(1 for c in self.cells if c.level == ResolutionLevel.FINE)
        med_cells = sum(1 for c in self.cells if c.level == ResolutionLevel.MEDIUM)
        coarse_cells = sum(1 for c in self.cells if c.level == ResolutionLevel.COARSE)
        refined_cells = sum(1 for c in self.cells if c.is_refined)

        # Equivalent cells if entire map had been constructed at uniform fine resolution
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        eq_fine_cells = int(np.round((width / self.fine_size) * (height / self.fine_size)))

        savings_ratio = 1.0 - (total_cells / eq_fine_cells) if eq_fine_cells > 0 else 0.0

        return {
            "type": "AdaptiveElevationGrid25D",
            "base_resolution_m": self.fine_size,
            "medium_resolution_m": self.medium_size,
            "coarse_resolution_m": self.coarse_size,
            "bounds": (self.min_x, self.max_x, self.min_y, self.max_y),
            "total_adaptive_cells": total_cells,
            "occupied_cells": occupied_cells,
            "empty_cells": empty_cells,
            "fine_cells_count": fine_cells,
            "medium_cells_count": med_cells,
            "coarse_cells_count": coarse_cells,
            "refined_cells_count": refined_cells,
            "fine_percentage": (fine_cells / total_cells * 100.0) if total_cells > 0 else 0.0,
            "medium_percentage": (med_cells / total_cells * 100.0) if total_cells > 0 else 0.0,
            "coarse_percentage": (coarse_cells / total_cells * 100.0) if total_cells > 0 else 0.0,
            "equivalent_uniform_fine_cells": eq_fine_cells,
            "cell_count_reduction_percentage": savings_ratio * 100.0,
        }

    def get_cell_at(self, x: float, y: float) -> Optional[AdaptiveCell25D]:
        """
        Locate the specific adaptive cell containing (X, Y).
        """
        for cell in self.cells:
            if cell.x_min <= x < cell.x_max and cell.y_min <= y < cell.y_max:
                return cell
        return None

    def rasterize_to_uniform_matrix(
        self,
        resolution: Optional[float] = None,
        attribute: str = "max_z"
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Rasterize the adaptive cells onto a uniform evaluation raster for direct comparison with fixed grids.
        """
        res = resolution if resolution is not None else self.fine_size
        nx = int(np.round((self.max_x - self.min_x) / res))
        ny = int(np.round((self.max_y - self.min_y) / res))

        matrix = np.full((ny, nx), self.empty_value, dtype=np.float64)

        for cell in self.cells:
            val = getattr(cell, attribute, self.empty_value)
            c_start = int(np.floor((cell.x_min - self.min_x) / res))
            c_end = int(np.ceil((cell.x_max - self.min_x) / res))
            r_start = int(np.floor((cell.y_min - self.min_y) / res))
            r_end = int(np.ceil((cell.y_max - self.min_y) / res))

            c_start = max(0, min(nx, c_start))
            c_end = max(0, min(nx, c_end))
            r_start = max(0, min(ny, r_start))
            r_end = max(0, min(ny, r_end))

            matrix[r_start:r_end, c_start:c_end] = val

        return matrix, (self.min_x, self.max_x, self.min_y, self.max_y)
