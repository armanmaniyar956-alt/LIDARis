"""
visualizer.py
=============
Visualization utilities for the LIDARis project.

Capabilities:
- Interactive 3D point cloud visualization using Open3D with elevation coloring.
- 2D orthographic top-down point cloud projection using Matplotlib.
- 2D fixed-resolution 2.5D elevation grid plotting.
- 2D true adaptive variable-resolution grid plotting using rectangular geometric patches.
- Spatial resolution allocation map (Fine vs Medium vs Coarse regions).
- Side-by-side and multi-panel comprehensive evaluation visualizations.
"""

from pathlib import Path
from typing import Optional, Union, Tuple, Any, Dict, List
import copy
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from src.pointcloud_io import numpy_to_o3d


def colorize_by_elevation(
    points: np.ndarray,
    colormap_name: str = "viridis"
) -> np.ndarray:
    """
    Generate RGB color values for an (N, 3) point cloud mapped to its Z (elevation) values.
    """
    z = points[:, 2]
    z_min, z_max = np.nanmin(z), np.nanmax(z)
    
    if np.isclose(z_max, z_min):
        normalized_z = np.zeros_like(z)
    else:
        normalized_z = (z - z_min) / (z_max - z_min)

    try:
        cmap = plt.get_cmap(colormap_name)
    except AttributeError:
        cmap = cm.get_cmap(colormap_name)

    rgba = cmap(normalized_z)
    rgb = rgba[:, :3].astype(np.float64)
    return rgb


def display_point_cloud_open3d(
    points_or_pcd: Union[np.ndarray, o3d.geometry.PointCloud],
    window_name: str = "LIDARis - 3D LiDAR Viewer",
    color_by_elevation: bool = True,
    point_size: float = 3.0,
    width: int = 1024,
    height: int = 768
) -> bool:
    """
    Open an interactive 3D viewer window using Open3D.
    """
    if isinstance(points_or_pcd, np.ndarray):
        colors = colorize_by_elevation(points_or_pcd) if color_by_elevation else None
        pcd = numpy_to_o3d(points_or_pcd, colors=colors)
    elif isinstance(points_or_pcd, o3d.geometry.PointCloud):
        pcd = points_or_pcd
        if color_by_elevation:
            pts = np.asarray(pcd.points)
            colors = colorize_by_elevation(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        raise TypeError(f"Expected np.ndarray or o3d.geometry.PointCloud, got {type(points_or_pcd)}")

    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=width, height=height)
        vis.add_geometry(pcd)
        
        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.array([0.08, 0.08, 0.12])
        
        vis.run()
        vis.destroy_window()
        return True
    except Exception as e:
        print(f"[Warning] Open3D window could not be opened: {e}")
        return False


def render_point_cloud_topdown(
    points: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "LiDAR Top-Down View (X-Y Plane)",
    colormap_name: str = "viridis"
) -> str:
    """
    Render a 2D orthographic top-down projection (X vs Y, colored by Z elevation).
    """
    if output_path is None:
        output_path = Path("data/outputs/lidar_topdown.png")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=points[:, 2],
        s=1.5,
        cmap=colormap_name,
        alpha=0.75
    )
    cbar = fig.colorbar(sc, ax=ax, label="Elevation Z (meters)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("X coordinate (meters)")
    ax.set_ylabel("Y coordinate (meters)")
    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close(fig)

    return str(output_path.resolve())


def plot_fixed_elevation_grid(
    grid: Any,
    value_key: str = "max_z",
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    colormap_name: str = "viridis",
    background_color: str = "#12121e"
) -> str:
    """
    Plot a 2D raster heatmap of the FixedElevationGrid25D.
    """
    if not hasattr(grid, "is_fitted") or not grid.is_fitted:
        raise ValueError("Provided grid is not fitted.")

    if output_path is None:
        output_path = Path(f"data/outputs/fixed_grid_{value_key}.png")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if value_key == "elevation_range":
        matrix = grid.elevation_range
    else:
        matrix = getattr(grid, value_key, None)

    if matrix is None:
        raise ValueError(f"Grid does not contain metric '{value_key}'")

    extent = [grid.min_x, grid.max_x, grid.min_y, grid.max_y]

    base_cmap = copy.copy(plt.get_cmap(colormap_name))
    if hasattr(base_cmap, "with_extremes"):
        base_cmap = base_cmap.with_extremes(bad=background_color)
    else:
        base_cmap.set_bad(color=background_color)
    masked_matrix = np.ma.masked_invalid(matrix)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    im = ax.imshow(
        masked_matrix,
        origin="lower",
        extent=extent,
        cmap=base_cmap,
        interpolation="nearest"
    )

    label_map = {
        "max_z": "Max Elevation Z (m)",
        "min_z": "Min Elevation Z (m)",
        "mean_z": "Mean Elevation Z (m)",
        "var_z": "Elevation Variance (m²)",
        "elevation_range": "Obstacle Height ΔZ (m)",
        "point_count": "Point Density (count/cell)",
    }
    cbar_label = label_map.get(value_key, value_key)
    cbar = fig.colorbar(im, ax=ax, label=cbar_label)

    plot_title = title or f"Fixed 2.5D Grid ({value_key}) | Res = {grid.resolution}m"
    ax.set_title(plot_title, fontsize=11, fontweight="bold")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle=":", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close(fig)

    return str(output_path.resolve())


def plot_pointcloud_vs_fixed_grid(
    points: np.ndarray,
    grid: Any,
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Generate a side-by-side comparison between the raw LiDAR point cloud
    and the baseline fixed-resolution 2.5D elevation grid.
    """
    if output_path is None:
        output_path = Path("data/outputs/pointcloud_vs_fixed_grid.png")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Left: Point cloud top-down projection
    sc = ax1.scatter(
        points[:, 0], points[:, 1],
        c=points[:, 2],
        s=1.2,
        cmap="viridis",
        alpha=0.75
    )
    fig.colorbar(sc, ax=ax1, label="Point Elevation Z (m)")
    ax1.set_title(f"A. Source LiDAR Point Cloud ({len(points):,} points)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.set_aspect("equal", "box")
    ax1.set_xlim(grid.min_x, grid.max_x)
    ax1.set_ylim(grid.min_y, grid.max_y)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Right: Fixed 2.5D Elevation Grid
    extent = [grid.min_x, grid.max_x, grid.min_y, grid.max_y]
    cmap = copy.copy(plt.get_cmap("viridis"))
    if hasattr(cmap, "with_extremes"):
        cmap = cmap.with_extremes(bad="#12121e")
    else:
        cmap.set_bad(color="#12121e")
    masked_grid = np.ma.masked_invalid(grid.max_z)

    im = ax2.imshow(
        masked_grid,
        origin="lower",
        extent=extent,
        cmap=cmap,
        interpolation="nearest"
    )
    fig.colorbar(im, ax=ax2, label="Grid Max Elevation Z (m)")
    total = grid.num_cells_x * grid.num_cells_y
    occ = np.sum(grid.occupied_mask)
    ax2.set_title(
        f"B. Baseline Fixed 2.5D Grid (Res={grid.resolution}m | {occ:,}/{total:,} occupied)",
        fontsize=11, fontweight="bold"
    )
    ax2.set_xlabel("X (meters)")
    ax2.set_ylabel("Y (meters)")
    ax2.set_aspect("equal", "box")
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close(fig)

    return str(output_path.resolve())


def plot_adaptive_grid_patches(
    adaptive_grid: Any,
    attribute: str = "max_z",
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    colormap_name: str = "viridis",
    background_color: str = "#12121e"
) -> str:
    """
    Render true adaptive variable-resolution grid cells using rectangular geometric patches.
    Visualizes the actual differing spatial cell sizes (0.2m, 0.4m, 0.8m).
    """
    if output_path is None:
        output_path = Path(f"data/outputs/adaptive_grid_{attribute}.png")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.set_facecolor(background_color)

    # Collect occupied cell values to determine colormap range
    values = []
    for c in adaptive_grid.cells:
        if c.is_occupied:
            v = getattr(c, attribute, np.nan)
            if not np.isnan(v):
                values.append(v)

    if len(values) > 0:
        vmin, vmax = np.min(values), np.max(values)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    cmap = plt.get_cmap(colormap_name)

    patch_list = []
    color_list = []

    for c in adaptive_grid.cells:
        rect = patches.Rectangle(
            (c.x_min, c.y_min),
            c.cell_size,
            c.cell_size,
            linewidth=0.3,
            edgecolor="#2a2a3e" if not c.is_refined else "#ff4444"
        )
        patch_list.append(rect)

        if c.is_occupied:
            val = getattr(c, attribute, np.nan)
            norm_val = (val - vmin) / (vmax - vmin) if not np.isnan(val) else 0.0
            color_list.append(cmap(norm_val))
        else:
            color_list.append((0.07, 0.07, 0.12, 1.0))

    pc = PatchCollection(patch_list, facecolors=color_list, match_original=True)
    ax.add_collection(pc)

    # Colorbar mapping
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=f"Adaptive Cell {attribute}")

    ax.set_xlim(adaptive_grid.min_x, adaptive_grid.max_x)
    ax.set_ylim(adaptive_grid.min_y, adaptive_grid.max_y)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    metrics = adaptive_grid.get_summary_metrics()
    plot_title = title or (
        f"Adaptive Variable-Resolution 2.5D Map ({metrics['total_adaptive_cells']} cells)\n"
        f"Red outlines = Refined obstacles | Fine: {metrics['fine_percentage']:.1f}%, "
        f"Med: {metrics['medium_percentage']:.1f}%, Coarse: {metrics['coarse_percentage']:.1f}%"
    )
    ax.set_title(plot_title, fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.25, color="white")

    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close(fig)

    return str(output_path.resolve())


def plot_resolution_allocation_map(
    adaptive_grid: Any,
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Render a color-coded map showing resolution levels (Fine, Medium, Coarse)
    and obstacle-refinement boundaries across the environment.
    """
    if output_path is None:
        output_path = Path("data/outputs/resolution_allocation_map.png")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.set_facecolor("#101018")

    color_map = {
        "FINE": "#00E676",     # Vibrant Spring Green
        "MEDIUM": "#FFC107",   # Golden Amber
        "COARSE": "#2979FF",   # Vivid Royal Blue
    }

    patch_list = []
    facecolors = []

    for c in adaptive_grid.cells:
        is_ref = c.is_refined
        rect = patches.Rectangle(
            (c.x_min, c.y_min),
            c.cell_size,
            c.cell_size,
            linewidth=1.8 if is_ref else 0.4,
            edgecolor="#FF1744" if is_ref else "#101018"
        )
        patch_list.append(rect)
        base_color = color_map.get(c.level.value, "#2979FF")
        facecolors.append(base_color)

    pc = PatchCollection(patch_list, facecolors=facecolors, match_original=True)
    ax.add_collection(pc)

    # Large, highly readable legend for hackathon presentation
    legend_elements = [
        patches.Patch(facecolor="#00E676", edgecolor="#101018", label=f"FINE ({adaptive_grid.fine_size}m) — Near Zone & Obstacles"),
        patches.Patch(facecolor="#FFC107", edgecolor="#101018", label=f"MEDIUM ({adaptive_grid.medium_size}m) — Intermediate Range"),
        patches.Patch(facecolor="#2979FF", edgecolor="#101018", label=f"COARSE ({adaptive_grid.coarse_size}m) — Distant / Flat Ground"),
        patches.Patch(facecolor="none", edgecolor="#FF1744", linewidth=2.0, label="REFINED — High-Variance Obstacles"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        framealpha=0.92,
        fontsize=10.5,
        facecolor="#181a24",
        edgecolor="#3f455a",
        labelcolor="white"
    )

    ax.set_xlim(adaptive_grid.min_x, adaptive_grid.max_x)
    ax.set_ylim(adaptive_grid.min_y, adaptive_grid.max_y)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    metrics = adaptive_grid.get_summary_metrics()
    plot_title = title or (
        f"Adaptive Resolution Allocation (SIH26053)\n"
        f"Fine: {metrics['fine_cells_count']} | Medium: {metrics['medium_cells_count']} | "
        f"Coarse: {metrics['coarse_cells_count']} | Refined: {metrics['refined_cells_count']}"
    )
    ax.set_title(plot_title, fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.2, color="white")

    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close(fig)

    return str(output_path.resolve())


def plot_full_sih_comparison(
    points: np.ndarray,
    fixed_grid: Any,
    adaptive_grid: Any,
    eval_results: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Generate a master 5-panel figure comprehensively comparing:
    1. Raw Point Cloud
    2. Baseline Fixed-Resolution 2.5D Grid
    3. Proposed Adaptive Variable-Resolution 2.5D Grid
    4. Spatial Resolution Allocation Map
    5. Quantitative Workload & Metric Evaluation Bar Chart
    """
    if output_path is None:
        output_path = Path("data/outputs/sih_comprehensive_comparison.png")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10), dpi=150)
    gs = fig.add_gridspec(2, 3)

    extent = [fixed_grid.min_x, fixed_grid.max_x, fixed_grid.min_y, fixed_grid.max_y]

    # Panel 1: Original Point Cloud
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=1.0, cmap="viridis", alpha=0.7)
    fig.colorbar(sc, ax=ax1, label="Z (m)")
    ax1.set_title(f"1. Source LiDAR Point Cloud\n({len(points):,} points)", fontsize=10, fontweight="bold")
    ax1.set_aspect("equal", "box")
    ax1.set_xlim(extent[0], extent[1])
    ax1.set_ylim(extent[2], extent[3])
    ax1.grid(True, linestyle=":", alpha=0.3)

    # Panel 2: Baseline Fixed Grid
    ax2 = fig.add_subplot(gs[0, 1])
    cmap_fixed = copy.copy(plt.get_cmap("viridis"))
    if hasattr(cmap_fixed, "with_extremes"):
        cmap_fixed = cmap_fixed.with_extremes(bad="#101018")
    else:
        cmap_fixed.set_bad(color="#101018")
    im2 = ax2.imshow(
        np.ma.masked_invalid(fixed_grid.max_z),
        origin="lower",
        extent=extent,
        cmap=cmap_fixed,
        interpolation="nearest"
    )
    fig.colorbar(im2, ax=ax2, label="Max Z (m)")
    f_tot = eval_results["fixed"]["total_cells"]
    f_occ = eval_results["fixed"]["occupied_cells"]
    ax2.set_title(
        f"2. Baseline Fixed 2.5D Grid\n(Res={fixed_grid.resolution}m | {f_occ:,}/{f_tot:,} cells)",
        fontsize=10, fontweight="bold"
    )
    ax2.set_aspect("equal", "box")
    ax2.grid(True, linestyle=":", alpha=0.3)

    # Panel 3: Adaptive Variable-Resolution Map
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#101018")
    vmin, vmax = np.nanmin(points[:, 2]), np.nanmax(points[:, 2])
    cmap_adapt = plt.get_cmap("viridis")
    patches_adapt = []
    colors_adapt = []
    for c in adaptive_grid.cells:
        rect = patches.Rectangle((c.x_min, c.y_min), c.cell_size, c.cell_size, linewidth=0.3, edgecolor="#2a2a3e")
        patches_adapt.append(rect)
        if c.is_occupied and not np.isnan(c.max_z):
            norm_z = (c.max_z - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            colors_adapt.append(cmap_adapt(norm_z))
        else:
            colors_adapt.append((0.07, 0.07, 0.12, 1.0))
    pc3 = PatchCollection(patches_adapt, facecolors=colors_adapt, match_original=True)
    ax3.add_collection(pc3)
    sm3 = plt.cm.ScalarMappable(cmap=cmap_adapt, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm3.set_array([])
    fig.colorbar(sm3, ax=ax3, label="Max Z (m)")
    a_tot = eval_results["adaptive"]["total_cells"]
    reduction = eval_results["comparison"]["cell_reduction_percentage"]
    ax3.set_title(
        f"3. Adaptive Variable-Res 2.5D Map\n({a_tot:,} cells | {reduction:.1f}% reduction)",
        fontsize=10, fontweight="bold"
    )
    ax3.set_aspect("equal", "box")
    ax3.set_xlim(extent[0], extent[1])
    ax3.set_ylim(extent[2], extent[3])
    ax3.grid(True, linestyle=":", alpha=0.2)

    # Panel 4: Resolution Allocation Map
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.set_facecolor("#101018")
    color_map = {"FINE": "#00E676", "MEDIUM": "#FFC107", "COARSE": "#2979FF"}
    res_patches = []
    res_colors = []
    for c in adaptive_grid.cells:
        is_ref = c.is_refined
        rect = patches.Rectangle(
            (c.x_min, c.y_min),
            c.cell_size,
            c.cell_size,
            linewidth=1.5 if is_ref else 0.4,
            edgecolor="#FF1744" if is_ref else "#101018"
        )
        res_patches.append(rect)
        res_colors.append(color_map.get(c.level.value, "#2979FF"))
    pc4 = PatchCollection(res_patches, facecolors=res_colors, match_original=True)
    ax4.add_collection(pc4)
    legend_elements = [
        patches.Patch(facecolor="#00E676", label=f"FINE ({adaptive_grid.fine_size}m) [{eval_results['adaptive']['fine_pct']:.1f}%]"),
        patches.Patch(facecolor="#FFC107", label=f"MEDIUM ({adaptive_grid.medium_size}m) [{eval_results['adaptive']['medium_pct']:.1f}%]"),
        patches.Patch(facecolor="#2979FF", label=f"COARSE ({adaptive_grid.coarse_size}m) [{eval_results['adaptive']['coarse_pct']:.1f}%]"),
        patches.Patch(facecolor="none", edgecolor="#FF1744", linewidth=1.8, label="Refined Obstacles"),
    ]
    ax4.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=9.5, facecolor="#181a24", edgecolor="#3f455a", labelcolor="white")
    ax4.set_title(
        "4. Spatial Resolution Level Allocation (Near=Fine, Mid=Med, Far=Coarse, Obstacles=Refined)",
        fontsize=10, fontweight="bold"
    )
    ax4.set_aspect("equal", "box")
    ax4.set_xlim(extent[0], extent[1])
    ax4.set_ylim(extent[2], extent[3])
    ax4.grid(True, linestyle=":", alpha=0.2)

    # Panel 5: Workload & Empirical Benchmark Bar Chart
    ax5 = fig.add_subplot(gs[1, 2])
    categories = ["Total Cells", "Runtime (ms)", "Elevation RMSE (cm)"]
    fixed_vals = [
        eval_results["fixed"]["total_cells"] / 1000.0,
        eval_results["fixed"]["runtime_ms"],
        eval_results["fixed"]["elevation_rmse_m"] * 100.0,
    ]
    adaptive_vals = [
        eval_results["adaptive"]["total_cells"] / 1000.0,
        eval_results["adaptive"]["runtime_ms"],
        eval_results["adaptive"]["elevation_rmse_m"] * 100.0,
    ]

    x = np.arange(len(categories))
    width = 0.35

    b1 = ax5.bar(x - width/2, fixed_vals, width, label="Fixed Res (0.2m)", color="#3498db")
    b2 = ax5.bar(x + width/2, adaptive_vals, width, label="Adaptive Res", color="#2ecc71")

    ax5.set_xticks(x)
    ax5.set_xticklabels(categories, fontsize=9)
    ax5.set_ylabel("Measured Value (cells in k, ms, cm)")
    ax5.set_title(
        f"5. Measured Benchmark Comparison\nCell Reduction: {reduction:.1f}%",
        fontsize=10, fontweight="bold"
    )
    ax5.legend(loc="upper right", fontsize=9)
    ax5.grid(True, linestyle="--", alpha=0.3)

    # Annotate bar heights
    for b in b1:
        h = b.get_height()
        ax5.annotate(f"{h:.1f}", (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=8)
    for b in b2:
        h = b.get_height()
        ax5.annotate(f"{h:.1f}", (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close(fig)

    return str(output_path.resolve())
