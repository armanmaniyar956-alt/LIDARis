# LIDARis 📡

**Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception**  
*Smart India Hackathon Prototype | Problem Statement ID: SIH26053*

---

## 📖 1. Project Purpose & Innovation

Robots, autonomous ground vehicles (AGVs), and drones rely on 3D LiDAR sensors to perceive dynamic outdoor and indoor environments. However, traditional mapping approaches present severe computational trade-offs:
* **Full 3D Voxel Grids** require massive amounts of memory and processing power, making real-time embedded navigation slow.
* **Uniform Fixed-Resolution 2.5D Grids** assign the exact same fine cell size (e.g., $0.2\text{ m}$) across the entire world, wasting computational cycles and memory on empty space, flat roads, and distant regions where laser beams diverge.

### The Core Innovation of LIDARis
**LIDARis** implements an **Adaptive Variable-Resolution 2.5D Elevation Grid Engine**. It dynamically allocates:
1. **FINE resolution ($0.2\text{ m}$)**: In close proximity to the robot ($d \le 6\text{ m}$) and in regions with high vertical complexity (tall obstacles, vehicles, pedestrians, barriers) regardless of distance.
2. **MEDIUM resolution ($0.4\text{ m}$)**: In intermediate zones ($6\text{ m} < d \le 12\text{ m}$) with moderate surface variation.
3. **COARSE resolution ($0.8\text{ m}$)**: In distant, flat, or empty zones ($d > 12\text{ m}$), reducing the number of cells by up to $16\times$ per macro-block.
4. **Dynamic Refinement**: When sequential frames indicate movement or when high elevation variation ($\Delta Z \ge 0.25\text{ m}$) is detected, coarse blocks are immediately subdivided into fine cells to guarantee safety-critical obstacle detection.

> [!NOTE]
> **True Spatial Representation:** Unlike systems that merely recolor fixed-grid pixels, LIDARis implements actual variable-sized spatial cells stored within the underlying data structures.

---

## 🏛️ 2. System Architecture

```text
LIDARis/
├── app/
│   ├── __init__.py
│   └── app.py                     # Interactive Streamlit Web Application
├── app.py                         # Root entrypoint for Streamlit
├── data/
│   ├── sample_data/               # Sample LiDAR files (.ply, .pcd, .bin)
│   └── outputs/                   # Generated evaluation plots and visual artifacts
├── src/
│   ├── __init__.py
│   ├── pointcloud_io.py           # Multi-format point cloud I/O & synthetic generator
│   ├── mapping_25d.py             # Baseline Fixed-Resolution 2.5D Elevation Grid
│   ├── adaptive_resolution.py     # Core Adaptive Variable-Resolution 2.5D Engine
│   ├── evaluation.py              # Empirical evaluation (cells, runtime, RMSE, memory)
│   ├── segmentation_interface.py  # RANSAC+DBSCAN ground/obstacle segmenter & ML interface
│   └── visualizer.py              # Open3D 3D visualizer & Matplotlib patch renderers
├── tests/
│   ├── test_basic.py
│   ├── test_pointcloud_io.py
│   ├── test_mapping_25d.py
│   ├── test_adaptive_resolution.py
│   ├── test_evaluation.py
│   └── test_segmentation.py
├── run_phase1_demo.py             # Quick verification CLI script
├── run_benchmarks.py              # End-to-end benchmarking and artifact generator
├── requirements.txt               # Dependencies with version constraints
└── README.md                      # Complete system documentation
```

---

## 📊 3. Empirical Benchmark Results (Measured)

Evaluated on a 5,700-point LiDAR scene ($30\text{m} \times 30\text{m}$ area) with ground plane, vehicle, pedestrian, and wall obstacles:

| Performance Metric | Baseline Fixed 2.5D Grid | Proposed Adaptive 2.5D Grid | Measured Improvement |
| :--- | :--- | :--- | :--- |
| **Grid Cell Size** | $0.20\text{ m}$ (Uniform) | $0.20\text{ m}$ / $0.40\text{ m}$ / $0.80\text{ m}$ | Dynamic $1\times, 2\times, 4\times$ |
| **Total Grid Cells** | **22,500** cells | **5,920** cells | **73.7% reduction in cell workload** |
| **Occupied Cells** | 4,041 cells | 2,673 cells | Efficient spatial consolidation |
| **Unoccupied Cells** | 18,459 cells | 3,247 cells | **82.4% reduction in empty cell storage** |
| **Elevation RMSE** | $0.4394\text{ m}$ | $0.4398\text{ m}$ | **$\Delta = 0.04\text{ cm}$ (near-zero loss in fidelity)** |
| **Refined Obstacle Cells** | N/A | 688 cells | $100\%$ obstacle detail preserved |
| **Memory Footprint** | $813.0\text{ KB}$ | $277.5\text{ KB}$ | **$65.9\%$ memory savings** |
| **Execution Runtime** | $47.7\text{ ms}$ | $158.4\text{ ms}$ | Real-time $(> 6\text{ Hz})$ in pure Python |

---

## 🚀 4. Installation & Getting Started

### Prerequisites
* Windows 10/11, Linux, or macOS.
* Python 3.12 (64-bit).

### Setup Environment
```powershell
# 1. Navigate to project root
cd "c:\Users\zameer shaikh\Pictures\LIDARis"

# 2. Activate Python 3.12 virtual environment
.\venv\Scripts\Activate.ps1

# 3. Install required packages (if not already installed)
pip install -r requirements.txt
```

---

## 🧪 5. How to Run Tests

Run the complete automated test suite (23 test cases):
```powershell
.\venv\Scripts\pytest.exe -v tests/
```
All tests verify:
- Point cloud I/O across `.ply`, `.pcd`, `.bin`, `.npy` formats.
- Fixed 2.5D grid statistics (`max_z`, `min_z`, `var_z`, `point_count`).
- Adaptive resolution allocation (Near=Fine, Far=Coarse, Obstacle=Refined).
- Temporal multi-frame dynamic change detection.
- Empirical metric calculation and RMSE verification.
- RANSAC ground extraction and DBSCAN obstacle clustering.

---

## 🖥️ 6. How to Launch the Streamlit Web Dashboard

Start the interactive engineering interface:
```powershell
.\venv\Scripts\streamlit.exe run app.py
```
Open your browser at `http://localhost:8501`.

### Key Features in the Dashboard:
1. **Interactive Controls:** Adjust base resolution, near/far distance thresholds, and height variance refinement sensitivity using sliders.
2. **Dataset Selector:** Test with the built-in synthetic benchmark, a 2-frame dynamic moving-obstacle simulation, or upload your own `.ply`, `.pcd`, or `.bin` files.
3. **Master SIH Evaluation Tab:** Displays side-by-side comparisons and live measured KPI metrics.
4. **Resolution Allocation Map:** Visualizes Fine (Green), Medium (Orange), Coarse (Gray), and Refined (White borders) regions.
5. **Explainability Inspector:** Enter any $(X, Y)$ coordinate to view the exact cell, its elevation statistics, and the human-readable explanation justifying why its resolution was chosen.
6. **Geometric Ground/Obstacle Segmentation:** Displays RANSAC ground terrain vs. DBSCAN obstacle clusters.

---

## 📈 7. Running the Offline Benchmark Script

To re-run benchmarks and generate all visualization images into `data/outputs/`:
```powershell
.\venv\Scripts\python.exe run_benchmarks.py
```

Generated images in `data/outputs/`:
* `sample_lidar_view.png` — Top-down orthographic projection of the raw point cloud.
* `pointcloud_vs_fixed_grid.png` — Comparison between point cloud and baseline fixed grid.
* `fixed_grid_max_z.png` & `fixed_grid_elevation_range.png` — Baseline elevation and obstacle clearance heatmaps.
* `adaptive_grid_max_z.png` — True variable-resolution 2.5D map showing variable cell sizes.
* `resolution_allocation_map.png` — Spatial distribution of Fine, Medium, and Coarse regions.
* `sih_comprehensive_comparison.png` — Master 5-panel SIH verification figure.
* `segmentation_baseline.png` — RANSAC + DBSCAN ground/obstacle classification.

---

## ⚠️ 8. Current Limitations & Future Work

### Genuine Current Capabilities
* Fully functional multi-format LiDAR loader (`.ply`, `.pcd`, `.bin`, `.npy`).
* Mathematically genuine adaptive variable-resolution 2.5D spatial data structures.
* Deterministic, explainable decision logic based on sensor range, elevation variance, and temporal change.
* Verified geometric segmentation using Open3D RANSAC and DBSCAN.
* Live interactive Streamlit dashboard.

### Limitations & Future Work (Designated for Phase 7 Extension)
1. **Deep Learning Segmentation:** A clean architectural interface (`BaseLiDARSegmenter`) is implemented. Integrating heavy pretrained deep networks (e.g., RandLA-Net or Cylinder3D) is reserved as future work once GPU acceleration and specific pretrained weights are configured.
2. **C++ / CUDA Acceleration:** The current prototype is implemented in optimized NumPy/Python. Translating the adaptive tree partitioning into C++ / CUDA kernels will achieve $>50\text{ FPS}$ on embedded robotics hardware (e.g., NVIDIA Jetson).
