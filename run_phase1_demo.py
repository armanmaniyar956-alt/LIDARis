"""
run_phase1_demo.py
==================
Demonstration & Verification script for Phase 1 of LIDARis.

Usage:
    # 1. Generate & inspect fallback synthetic LiDAR data and save 2D projection:
    python run_phase1_demo.py

    # 2. Launch interactive Open3D 3D visualizer:
    python run_phase1_demo.py --view

    # 3. Test loading a specific file (.ply, .pcd, or .bin):
    python run_phase1_demo.py --file data/sample_data/synthetic_scene.pcd --view
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Ensure UTF-8 output encoding for Windows PowerShell / CMD
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.pointcloud_io import (
    load_point_cloud,
    save_point_cloud,
    generate_synthetic_lidar_scene,
    get_or_create_fallback_pointcloud,
)
from src.visualizer import display_point_cloud_open3d, render_point_cloud_topdown


def main():
    parser = argparse.ArgumentParser(
        description="LIDARis Phase 1: Point Cloud Ingestion & Visualization Verification"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to point cloud file (.ply, .pcd, .bin, .npy). If omitted, fallback data is used."
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open an interactive 3D Open3D window to inspect the point cloud."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("[LIDARis] Phase 1 Verification")
    print("=" * 60)

    # 1. Determine input file
    sample_dir = Path("data/sample_data")
    sample_dir.mkdir(parents=True, exist_ok=True)
    fallback_ply = sample_dir / "synthetic_scene.ply"
    fallback_pcd = sample_dir / "synthetic_scene.pcd"
    fallback_bin = sample_dir / "synthetic_scene.bin"

    # Pre-generate sample files in all 3 formats for easy user testing
    if not fallback_ply.exists() or not fallback_pcd.exists() or not fallback_bin.exists():
        print("[1/4] Generating initial fallback synthetic LiDAR dataset (.ply, .pcd, .bin)...")
        synthetic_pts = generate_synthetic_lidar_scene(seed=42)
        save_point_cloud(fallback_ply, synthetic_pts)
        save_point_cloud(fallback_pcd, synthetic_pts)
        save_point_cloud(fallback_bin, synthetic_pts)
        print(f"      Saved: {fallback_ply}")
        print(f"      Saved: {fallback_pcd}")
        print(f"      Saved: {fallback_bin}")

    target_file = Path(args.file) if args.file else fallback_ply
    print(f"\n[2/4] Loading point cloud from: {target_file}")
    points = load_point_cloud(target_file)

    # 2. Display measured characteristics (no unmeasured claims)
    print("\n[3/4] Point Cloud Inspection:")
    print(f"      - Total 3D points loaded : {len(points):,}")
    print(f"      - Array shape            : {points.shape}")
    print(f"      - Coordinate data type   : {points.dtype}")
    print(f"      - X range (width)        : [{points[:, 0].min():.2f}m, {points[:, 0].max():.2f}m]")
    print(f"      - Y range (depth)        : [{points[:, 1].min():.2f}m, {points[:, 1].max():.2f}m]")
    print(f"      - Z range (elevation)    : [{points[:, 2].min():.2f}m, {points[:, 2].max():.2f}m]")

    # 3. Render 2D top-down projection to verify geometry
    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    topdown_img = output_dir / "sample_lidar_view.png"
    render_point_cloud_topdown(points, output_path=topdown_img)
    print(f"\n[4/4] Rendered 2D orthographic top-down map to:\n      {topdown_img.resolve()}")

    # 4. Open3D 3D Display
    if args.view:
        print("\nOpening Open3D interactive viewer (close window when finished)...")
        display_point_cloud_open3d(points, window_name=f"LIDARis - {target_file.name}")
        print("Viewer window closed successfully.")
    else:
        print("\nTip: Run with '--view' to open the interactive 3D Open3D viewer:")
        print(f"   .\\venv\\Scripts\\python.exe run_phase1_demo.py --view")

    print("\n" + "=" * 60)
    print("[SUCCESS] Phase 1 Verification Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
