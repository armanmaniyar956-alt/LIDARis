"""
run_benchmarks.py
=================
Executes full end-to-end benchmark evaluation of the LIDARis software pipeline.
Measures real performance metrics and generates all visual output artifacts.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output encoding for Windows PowerShell / CMD
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

from src.pointcloud_io import (
    get_or_create_fallback_pointcloud,
    generate_synthetic_lidar_scene,
)
from src.evaluation import evaluate_fixed_vs_adaptive
from src.segmentation_interface import GeometricGroundObstacleSegmenter, SemanticClass
from src.visualizer import (
    plot_pointcloud_vs_fixed_grid,
    plot_fixed_elevation_grid,
    plot_adaptive_grid_patches,
    plot_resolution_allocation_map,
    plot_full_sih_comparison,
    render_point_cloud_topdown,
)
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def main():
    print("=" * 70)
    print("🚀 LIDARis (SIH26053): End-to-End Benchmark & Verification Pipeline")
    print("=" * 70)

    # 1. Load data
    print("\n[Step 1/5] Ingesting Fallback Synthetic LiDAR Point Cloud...")
    points = get_or_create_fallback_pointcloud()
    print(f"           Loaded {len(points):,} 3D points.")
    print(f"           Bounds: X=[{points[:,0].min():.2f}m, {points[:,0].max():.2f}m], "
          f"Y=[{points[:,1].min():.2f}m, {points[:,1].max():.2f}m], "
          f"Z=[{points[:,2].min():.2f}m, {points[:,2].max():.2f}m]")

    # 2. Run Evaluation
    print("\n[Step 2/5] Running Empirical Benchmark (Fixed vs Adaptive Mapping)...")
    results = evaluate_fixed_vs_adaptive(
        points=points,
        base_resolution=0.20,
        multipliers=(1, 2, 4),
        near_distance_threshold=6.0,
        far_distance_threshold=12.0,
        height_diff_threshold=0.25
    )

    fixed = results["fixed"]
    adapt = results["adaptive"]
    comp = results["comparison"]

    print("\n" + "-" * 70)
    print(f"{'METRIC':<35} | {'FIXED 2.5D GRID':<16} | {'ADAPTIVE 2.5D GRID':<16}")
    print("-" * 70)
    print(f"{'Base Resolution':<35} | {fixed['resolution_m']:<16.2f} | {adapt['fine_resolution_m']:<16.2f}")
    print(f"{'Total Grid Cells':<35} | {fixed['total_cells']:<16,d} | {adapt['total_cells']:<16,d}")
    print(f"{'Occupied Cells':<35} | {fixed['occupied_cells']:<16,d} | {adapt['occupied_cells']:<16,d}")
    print(f"{'Empty Cells':<35} | {fixed['empty_cells']:<16,d} | {adapt['empty_cells']:<16,d}")
    print(f"{'Cell Reduction Percentage':<35} | {'0.0% (Baseline)':<16} | {comp['cell_reduction_percentage']:<15.1f}%")
    print(f"{'FINE Cells (0.2m)':<35} | {'N/A':<16} | {adapt['fine_cells_count']:<8,d} ({adapt['fine_pct']:.1f}%)")
    print(f"{'MEDIUM Cells (0.4m)':<35} | {'N/A':<16} | {adapt['medium_cells_count']:<8,d} ({adapt['medium_pct']:.1f}%)")
    print(f"{'COARSE Cells (0.8m)':<35} | {'N/A':<16} | {adapt['coarse_cells_count']:<8,d} ({adapt['coarse_pct']:.1f}%)")
    print(f"{'Refined Obstacle Cells':<35} | {'N/A':<16} | {adapt['refined_cells_count']:<16,d}")
    print(f"{'Execution Runtime (ms)':<35} | {fixed['runtime_ms']:<16.2f} | {adapt['runtime_ms']:<16.2f}")
    print(f"{'Elevation RMSE Error (m)':<35} | {fixed['elevation_rmse_m']:<16.4f} | {adapt['elevation_rmse_m']:<16.4f}")
    print(f"{'Approx Memory Footprint (KB)':<35} | {fixed['approx_memory_kb']:<16.1f} | {adapt['approx_memory_kb']:<16.1f}")
    print("-" * 70)

    # 3. Generate Visual Artifacts
    print("\n[Step 3/5] Generating Visual Artifacts in data/outputs/...")
    out_dir = Path("data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = render_point_cloud_topdown(points, out_dir / "sample_lidar_view.png")
    print(f"           Saved: {Path(p1).name}")

    p2 = plot_pointcloud_vs_fixed_grid(points, fixed["grid_obj"], out_dir / "pointcloud_vs_fixed_grid.png")
    print(f"           Saved: {Path(p2).name}")

    p3 = plot_fixed_elevation_grid(fixed["grid_obj"], "max_z", output_path=out_dir / "fixed_grid_max_z.png")
    print(f"           Saved: {Path(p3).name}")

    p4 = plot_fixed_elevation_grid(fixed["grid_obj"], "elevation_range", output_path=out_dir / "fixed_grid_elevation_range.png")
    print(f"           Saved: {Path(p4).name}")

    p5 = plot_adaptive_grid_patches(adapt["grid_obj"], "max_z", output_path=out_dir / "adaptive_grid_max_z.png")
    print(f"           Saved: {Path(p5).name}")

    p6 = plot_resolution_allocation_map(adapt["grid_obj"], output_path=out_dir / "resolution_allocation_map.png")
    print(f"           Saved: {Path(p6).name}")

    p7 = plot_full_sih_comparison(points, fixed["grid_obj"], adapt["grid_obj"], results, out_dir / "sih_comprehensive_comparison.png")
    print(f"           Saved: {Path(p7).name}")

    # 4. Geometric Segmentation
    print("\n[Step 4/5] Running Phase 7 Geometric Segmentation (RANSAC + DBSCAN)...")
    segmenter = GeometricGroundObstacleSegmenter()
    labels = segmenter.segment(points)
    ground_n = int(np.sum(labels == SemanticClass.GROUND))
    obst_n = int(np.sum(labels == SemanticClass.STATIC_OBSTACLE))
    unlab_n = int(np.sum(labels == SemanticClass.UNLABELED))
    print(f"           Classified: {ground_n:,} ground points ({(ground_n/len(labels))*100:.1f}%), "
          f"{obst_n:,} obstacle points ({(obst_n/len(labels))*100:.1f}%), "
          f"{unlab_n:,} unlabeled points.")

    fig_seg, ax_seg = plt.subplots(figsize=(8, 6), dpi=150)
    ax_seg.set_facecolor("#101018")
    c_map = {SemanticClass.GROUND: "#2ecc71", SemanticClass.STATIC_OBSTACLE: "#e74c3c", SemanticClass.UNLABELED: "#95a5a6"}
    colors = [c_map.get(l, "#95a5a6") for l in labels]
    ax_seg.scatter(points[:, 0], points[:, 1], c=colors, s=1.5, alpha=0.8)
    leg = [
        patches.Patch(facecolor="#2ecc71", label=f"Ground ({ground_n:,})"),
        patches.Patch(facecolor="#e74c3c", label=f"Obstacles ({obst_n:,})"),
        patches.Patch(facecolor="#95a5a6", label=f"Unlabeled ({unlab_n:,})"),
    ]
    ax_seg.legend(handles=leg, loc="upper right")
    ax_seg.set_title("Geometric Ground vs. Obstacle Segmentation", fontsize=11, fontweight="bold")
    ax_seg.set_xlabel("X (m)")
    ax_seg.set_ylabel("Y (m)")
    ax_seg.set_aspect("equal", "box")
    ax_seg.grid(True, linestyle=":", alpha=0.3)
    p8 = out_dir / "segmentation_baseline.png"
    plt.tight_layout()
    plt.savefig(str(p8))
    plt.close(fig_seg)
    print(f"           Saved: {p8.name}")

    # 5. Summary
    print("\n[Step 5/5] All 8 visual artifacts generated and stored in data/outputs/.")
    print("=" * 70)
    print(f"✅ BENCHMARK COMPLETE: Achieved {comp['cell_reduction_percentage']:.1f}% grid cell workload reduction")
    print(f"   while preserving obstacle elevation RMSE within {comp['rmse_difference_m']*100:.2f} cm difference!")
    print("=" * 70)


if __name__ == "__main__":
    main()
