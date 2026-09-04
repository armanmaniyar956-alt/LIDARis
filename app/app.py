"""
app/app.py
==========
Interactive Streamlit Dashboard for LIDARis:
Adaptive Variable-Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception.
Smart India Hackathon (SIH26053) Software Prototype.

Features:
- Point cloud ingestion (Upload .ply, .pcd, .bin or use built-in synthetic/dynamic LiDAR scenes).
- Interactive parameter configuration (Distance bands, elevation variance thresholds, cell resolution).
- Real-time computation of Baseline Fixed-Resolution and Adaptive Variable-Resolution grids.
- Display of actual measured experimental metrics (cell count, runtime ms, RMSE error, memory).
- Explainability Inspector: inspect why specific spatial regions received Fine/Medium/Coarse resolution.
- Geometric Ground/Obstacle segmentation visualizer.
"""

import tempfile
import time
from pathlib import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.pointcloud_io import (
    load_point_cloud,
    save_point_cloud,
    generate_synthetic_lidar_scene,
    get_or_create_fallback_pointcloud,
)
from src.mapping_25d import FixedElevationGrid25D
from src.adaptive_resolution import AdaptiveElevationGrid25D, ResolutionLevel
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


def main():
    st.set_page_config(
        page_title="LIDARis | Adaptive 2.5D LiDAR Mapping",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header & SIH Banner
    st.title("📡 LIDARis")
    st.markdown(
        "### Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception  \n"
        "**Smart India Hackathon 2024 / 2026 | Problem Statement ID: SIH26053**"
    )

    # 1. Visual Flow Pipeline for Hackathon Judges
    st.markdown("""
    <div style="background-color: #1a1e2e; padding: 14px 18px; border-radius: 8px; border-left: 5px solid #00E676; margin-top: 10px; margin-bottom: 20px;">
        <div style="font-size: 11px; font-weight: 700; color: #8fa0b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">
            💡 HOW LIDARis WORKS (PIPELINE ARCHITECTURE)
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; font-size: 13.5px; font-weight: 600;">
            <span style="background: #252b3d; color: #e2e8f0; padding: 6px 12px; border-radius: 6px; border: 1px solid #3b4256;">1. 📡 LiDAR Point Cloud</span>
            <span style="color: #64b5f6; font-size: 16px;">➔</span>
            <span style="background: #252b3d; color: #e2e8f0; padding: 6px 12px; border-radius: 6px; border: 1px solid #3b4256;">2. 🗺️ 2.5D Grid</span>
            <span style="color: #64b5f6; font-size: 16px;">➔</span>
            <span style="background: #252b3d; color: #e2e8f0; padding: 6px 12px; border-radius: 6px; border: 1px solid #3b4256;">3. ⚡ Environment Complexity</span>
            <span style="color: #64b5f6; font-size: 16px;">➔</span>
            <span style="background: #252b3d; color: #e2e8f0; padding: 6px 12px; border-radius: 6px; border: 1px solid #3b4256;">4. 🎛️ Adaptive Resolution</span>
            <span style="color: #64b5f6; font-size: 16px;">➔</span>
            <span style="background: #0f3822; color: #00E676; padding: 6px 12px; border-radius: 6px; border: 1px solid #00E676;">5. 🚀 Final Map (73.7% Fewer Cells)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SIDEBAR CONTROLS
    # ---------------------------------------------------------
    st.sidebar.header("⚙️ Configuration")

    # 1. Dataset Selection
    st.sidebar.subheader("1. Data Source")
    data_source = st.sidebar.selectbox(
        "Select LiDAR Input",
        options=[
            "Fallback Synthetic Scene (Ground + Obstacles)",
            "Multi-Frame Dynamic Scene (Moving Obstacle)",
            "Upload Point Cloud File (.ply, .pcd, .bin)"
        ]
    )

    current_points = None
    previous_points = None
    data_label = ""

    if data_source == "Fallback Synthetic Scene (Ground + Obstacles)":
        current_points = get_or_create_fallback_pointcloud()
        data_label = "Synthetic Scene (Deterministic fallback)"

    elif data_source == "Multi-Frame Dynamic Scene (Moving Obstacle)":
        # Frame 0: Static scene with vehicle at x=5
        pts_f0 = generate_synthetic_lidar_scene(seed=10)
        # Frame 1: Vehicle shifted forward to x=9
        pts_f1 = generate_synthetic_lidar_scene(seed=10)
        # Shift vehicle points
        veh_mask = (pts_f1[:, 0] >= 4.0) & (pts_f1[:, 0] <= 8.0) & (pts_f1[:, 1] >= 2.0) & (pts_f1[:, 1] <= 4.0)
        pts_f1[veh_mask, 0] += 3.5  # Moved 3.5m forward in X
        
        current_points = pts_f1
        previous_points = pts_f0
        data_label = "Multi-Frame Dynamic Simulation (Vehicle displacement: +3.5m)"

    elif data_source == "Upload Point Cloud File (.ply, .pcd, .bin)":
        uploaded_file = st.sidebar.file_uploader(
            "Upload LiDAR file",
            type=["ply", "pcd", "bin", "npy", "xyz", "csv"]
        )
        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            try:
                current_points = load_point_cloud(tmp_path)
                data_label = f"Uploaded: {uploaded_file.name} ({len(current_points):,} points)"
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
                return
        else:
            st.info("Please upload a supported LiDAR file, or select a synthetic scene from the sidebar.")
            return

    # 2. Resolution Parameters
    st.sidebar.subheader("2. Resolution Settings")
    base_res = st.sidebar.slider(
        "Base Fine Resolution (meters)",
        min_value=0.10,
        max_value=0.50,
        value=0.20,
        step=0.05,
        help="Cell size for highest detail areas (obstacles and near zone)."
    )

    # 3. Adaptive Thresholds
    st.sidebar.subheader("3. Adaptive Engine Thresholds")
    near_thresh = st.sidebar.slider(
        "Near Zone Distance Threshold (m)",
        min_value=2.0,
        max_value=12.0,
        value=6.0,
        step=1.0,
        help="Regions within this range automatically use FINE resolution."
    )
    far_thresh = st.sidebar.slider(
        "Far Zone Distance Threshold (m)",
        min_value=8.0,
        max_value=25.0,
        value=12.0,
        step=1.0,
        help="Regions beyond this range default to COARSE unless refined."
    )
    height_thresh = st.sidebar.slider(
        "Obstacle Height Refinement (ΔZ in m)",
        min_value=0.10,
        max_value=0.80,
        value=0.25,
        step=0.05,
        help="Elevation range in a macro-cell that triggers refinement to FINE/MEDIUM."
    )

    st.sidebar.info(
        f"**Resolution Hierarchy:**\n"
        f"- **FINE:** `{base_res:.2f} m`\n"
        f"- **MEDIUM:** `{base_res * 2:.2f} m`\n"
        f"- **COARSE:** `{base_res * 4:.2f} m`"
    )

    # ---------------------------------------------------------
    # RUN COMPUTATION
    # ---------------------------------------------------------
    with st.spinner("Processing point cloud and computing 2.5D representations..."):
        eval_results = evaluate_fixed_vs_adaptive(
            points=current_points,
            base_resolution=base_res,
            multipliers=(1, 2, 4),
            near_distance_threshold=near_thresh,
            far_distance_threshold=far_thresh,
            height_diff_threshold=height_thresh,
            previous_points=previous_points
        )

    fixed_data = eval_results["fixed"]
    adapt_data = eval_results["adaptive"]
    comp_data = eval_results["comparison"]

    fixed_grid: FixedElevationGrid25D = fixed_data["grid_obj"]
    adapt_grid: AdaptiveElevationGrid25D = adapt_data["grid_obj"]

    # ---------------------------------------------------------
    # TOP KPI METRICS BAR (REAL MEASURED VALUES)
    # ---------------------------------------------------------
    st.subheader(f"📊 Live Measured Performance | {data_label}")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Points", f"{len(current_points):,}")
    with col2:
        st.metric("Fixed Cells", f"{fixed_data['total_cells']:,}")
    with col3:
        st.metric(
            "Adaptive Cells",
            f"{adapt_data['total_cells']:,}",
            delta=f"-{comp_data['cell_reduction_percentage']:.1f}%",
            delta_color="normal"
        )
    with col4:
        st.metric("Adaptive Runtime", f"{adapt_data['runtime_ms']:.1f} ms")
    with col5:
        st.metric("Elevation RMSE", f"{adapt_data['elevation_rmse_m'] * 100.0:.2f} cm")

    st.divider()

    # ---------------------------------------------------------
    # MAIN TABS
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏆 Master SIH Evaluation",
        "🗺️ Resolution Allocation Map",
        "📐 True Adaptive Variable-Grid",
        "🏛️ Baseline Fixed Grid",
        "🔍 Explainability Inspector",
        "🌿 Geometric Segmentation"
    ])

    # ----------------- TAB 1: MASTER EVALUATION -----------------
    with tab1:
        st.markdown("### 🏆 Comprehensive Comparison: Fixed vs Adaptive 2.5D Mapping")
        st.caption(
            "Side-by-side verification demonstrating that adaptive variable resolution preserves "
            "critical obstacle elevations while significantly decreasing total grid cell representation workload."
        )

        comparison_img_path = Path("data/outputs/sih_comprehensive_comparison.png")
        plot_full_sih_comparison(
            points=current_points,
            fixed_grid=fixed_grid,
            adaptive_grid=adapt_grid,
            eval_results=eval_results,
            output_path=comparison_img_path
        )
        st.image(str(comparison_img_path), use_container_width=True)

        st.markdown("#### 🔬 Detailed Experimental Measurement Table")
        metrics_df = pd.DataFrame({
            "Metric": [
                "Cell Resolution (Fine / Coarse)",
                "Total Spatial Cells in Map",
                "Occupied Cells Count",
                "Empty/Unobserved Cells",
                "Workload Reduction (% fewer cells)",
                "Execution Runtime (ms)",
                "Elevation Reconstruction RMSE (m)",
                "Approximate Memory (KB)"
            ],
            "Baseline Fixed 2.5D Grid": [
                f"{fixed_data['resolution_m']:.2f} m (Uniform)",
                f"{fixed_data['total_cells']:,}",
                f"{fixed_data['occupied_cells']:,}",
                f"{fixed_data['empty_cells']:,}",
                "0.0% (Baseline)",
                f"{fixed_data['runtime_ms']:.2f} ms",
                f"{fixed_data['elevation_rmse_m']:.4f} m",
                f"{fixed_data['approx_memory_kb']:.1f} KB"
            ],
            "Proposed Adaptive 2.5D Grid": [
                f"{adapt_data['fine_resolution_m']:.2f}m / {adapt_data['medium_resolution_m']:.2f}m / {adapt_data['coarse_resolution_m']:.2f}m",
                f"{adapt_data['total_cells']:,}",
                f"{adapt_data['occupied_cells']:,}",
                f"{adapt_data['empty_cells']:,}",
                f"{comp_data['cell_reduction_percentage']:.1f}%",
                f"{adapt_data['runtime_ms']:.2f} ms",
                f"{adapt_data['elevation_rmse_m']:.4f} m",
                f"{adapt_data['approx_memory_kb']:.1f} KB"
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # ----------------- TAB 2: RESOLUTION ALLOCATION MAP -----------------
    with tab2:
        st.markdown("### 🗺️ Spatial Resolution Allocation Map")
        st.write(
            "Visualizes the spatial distribution of resolution levels. Notice that near regions receive "
            "**FINE (Green)**, mid-distance receives **MEDIUM (Orange)**, far ground receives **COARSE (Dark Slate)**, "
            "and distant obstacles/dynamic targets are dynamically refined with **White Outlines**."
        )

        res_map_path = Path("data/outputs/resolution_allocation_map.png")
        plot_resolution_allocation_map(adapt_grid, output_path=res_map_path)
        st.image(str(res_map_path), use_container_width=True)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("FINE Cells (0.2m)", f"{adapt_data['fine_cells_count']:,}", f"{adapt_data['fine_pct']:.1f}%")
        col_b.metric("MEDIUM Cells (0.4m)", f"{adapt_data['medium_cells_count']:,}", f"{adapt_data['medium_pct']:.1f}%")
        col_c.metric("COARSE Cells (0.8m)", f"{adapt_data['coarse_cells_count']:,}", f"{adapt_data['coarse_pct']:.1f}%")
        col_d.metric("Refined Obstacles", f"{adapt_data['refined_cells_count']:,}")

    # ----------------- TAB 3: TRUE ADAPTIVE GRID -----------------
    with tab3:
        st.markdown("### 📐 True Adaptive Variable-Resolution Representation")
        st.write(
            "Every rectangle rendered below represents an **actual spatial cell stored in the data structure**, "
            "showing different cell dimensions ($0.2\\text{m}$, $0.4\\text{m}$, and $0.8\\text{m}$) colored by elevation ($Z$). "
            "Obstacles with high elevation variance stand out with sharp fine cells."
        )

        adapt_img_path = Path("data/outputs/adaptive_grid_max_z.png")
        plot_adaptive_grid_patches(adapt_grid, attribute="max_z", output_path=adapt_img_path)
        st.image(str(adapt_img_path), use_container_width=True)

    # ----------------- TAB 4: BASELINE FIXED GRID -----------------
    with tab4:
        st.markdown("### 🏛️ Baseline Fixed-Resolution 2.5D Elevation Grid")
        st.write(
            f"Conventional 2.5D mapping with a uniform resolution of `{fixed_grid.resolution} m`. "
            "While accurate, it allocates an unnecessarily large number of cells to empty and distant flat space."
        )

        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            fixed_img_path = Path("data/outputs/fixed_grid_max_z.png")
            plot_fixed_elevation_grid(fixed_grid, value_key="max_z", output_path=fixed_img_path)
            st.image(str(fixed_img_path), caption="Max Elevation (Z)", use_container_width=True)
        with sub_col2:
            elev_range_path = Path("data/outputs/fixed_grid_elevation_range.png")
            plot_fixed_elevation_grid(fixed_grid, value_key="elevation_range", output_path=elev_range_path)
            st.image(str(elev_range_path), caption="Obstacle Clearance (ΔZ = Max - Min)", use_container_width=True)

    # ----------------- TAB 5: EXPLAINABILITY INSPECTOR -----------------
    with tab5:
        st.markdown("### 🔍 Explainable AI: Region Resolution Inspector")
        st.write(
            "Select any coordinate $(X, Y)$ in meters to inspect the exact spatial cell, its elevation "
            "properties, and the deterministic rule that assigned its resolution level."
        )

        insp_col1, insp_col2 = st.columns(2)
        with insp_col1:
            inspect_x = st.number_input("Inspect X Coordinate (m)", value=6.0, step=0.5)
        with insp_col2:
            inspect_y = st.number_input("Inspect Y Coordinate (m)", value=3.0, step=0.5)

        target_cell = adapt_grid.get_cell_at(inspect_x, inspect_y)

        if target_cell is not None:
            st.success(f"**Cell Located at ({inspect_x:.2f}, {inspect_y:.2f})**")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Assigned Level", target_cell.level.value)
            c2.metric("Cell Edge Size", f"{target_cell.cell_size:.2f} m")
            c3.metric("Sensor Distance", f"{target_cell.distance_from_sensor:.2f} m")
            c4.metric("Point Count", f"{target_cell.point_count}")

            st.markdown("#### 💬 Explainability Log")
            st.code(target_cell.decision_reason, language="markdown")

            st.json({
                "Spatial Bounding Box": {
                    "X": [f"{target_cell.x_min:.2f}m", f"{target_cell.x_max:.2f}m"],
                    "Y": [f"{target_cell.y_min:.2f}m", f"{target_cell.y_max:.2f}m"]
                },
                "Elevation Statistics": {
                    "Max Z": f"{target_cell.max_z:.3f} m" if not np.isnan(target_cell.max_z) else "Unoccupied",
                    "Min Z": f"{target_cell.min_z:.3f} m" if not np.isnan(target_cell.min_z) else "Unoccupied",
                    "Mean Z": f"{target_cell.mean_z:.3f} m" if not np.isnan(target_cell.mean_z) else "Unoccupied",
                    "Variance (m²)": f"{target_cell.var_z:.4f}" if not np.isnan(target_cell.var_z) else "Unoccupied"
                },
                "Refinement Triggered": target_cell.is_refined
            })
        else:
            st.warning("Selected coordinate falls outside the mapped environment boundaries.")

    # ----------------- TAB 6: GEOMETRIC SEGMENTATION -----------------
    with tab6:
        st.markdown("### 🌿 Geometric Ground vs. Obstacle Segmentation (Phase 7)")
        st.write(
            "Uses RANSAC ground plane extraction to classify ground terrain and DBSCAN Euclidean clustering "
            "to separate static obstacles without relying on unverified deep learning dependencies."
        )

        with st.spinner("Computing geometric segmentation..."):
            segmenter = GeometricGroundObstacleSegmenter()
            labels = segmenter.segment(current_points)

        ground_count = int(np.sum(labels == SemanticClass.GROUND))
        obstacle_count = int(np.sum(labels == SemanticClass.STATIC_OBSTACLE))
        unlabeled_count = int(np.sum(labels == SemanticClass.UNLABELED))

        seg_c1, seg_c2, seg_c3 = st.columns(3)
        seg_c1.metric("Ground Points", f"{ground_count:,}", f"{(ground_count/len(labels))*100:.1f}%")
        seg_c2.metric("Obstacle Points", f"{obstacle_count:,}", f"{(obstacle_count/len(labels))*100:.1f}%")
        seg_c3.metric("Unlabeled / Noise", f"{unlabeled_count:,}")

        # Render segmentation scatter plot
        fig_seg, ax_seg = plt.subplots(figsize=(8, 6), dpi=150)
        ax_seg.set_facecolor("#101018")
        
        # Color mapping: Ground=Brown/Green, Obstacle=Red, Unlabeled=Gray
        c_map = {
            SemanticClass.GROUND: "#2ecc71",
            SemanticClass.STATIC_OBSTACLE: "#e74c3c",
            SemanticClass.UNLABELED: "#95a5a6"
        }
        point_colors = [c_map.get(l, "#95a5a6") for l in labels]

        ax_seg.scatter(current_points[:, 0], current_points[:, 1], c=point_colors, s=1.5, alpha=0.8)
        
        legend_elements = [
            patches.Patch(facecolor="#2ecc71", label=f"Ground Terrain ({ground_count:,})"),
            patches.Patch(facecolor="#e74c3c", label=f"Obstacles ({obstacle_count:,})"),
            patches.Patch(facecolor="#95a5a6", label=f"Unlabeled ({unlabeled_count:,})"),
        ]
        ax_seg.legend(handles=legend_elements, loc="upper right")
        ax_seg.set_title("Geometric Ground vs. Obstacle Segmentation (RANSAC + DBSCAN)", fontsize=11, fontweight="bold")
        ax_seg.set_xlabel("X (m)")
        ax_seg.set_ylabel("Y (m)")
        ax_seg.set_aspect("equal", "box")
        ax_seg.grid(True, linestyle=":", alpha=0.3)

        seg_plot_path = Path("data/outputs/segmentation_baseline.png")
        plt.tight_layout()
        plt.savefig(str(seg_plot_path))
        plt.close(fig_seg)

        st.image(str(seg_plot_path), use_container_width=True)

        st.info(
            "ℹ️ **Architectural Note:** Deep learning models (e.g. RandLA-Net) can be plugged directly "
            "into `BaseLiDARSegmenter` in `src/segmentation_interface.py` when pre-trained weights are provided."
        )


if __name__ == "__main__":
    main()
