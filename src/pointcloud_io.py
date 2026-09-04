"""
pointcloud_io.py
================
LiDAR point cloud Input/Output module for the LIDARis project.

Capabilities:
- Load 3D point clouds from common formats: .ply, .pcd, .bin (KITTI LiDAR format), .npy, .xyz, .csv.
- Convert between NumPy arrays and Open3D PointCloud geometries.
- Save point clouds to disk (.ply, .pcd, .bin, .npy).
- Generate synthetic LiDAR test scenes as fallbacks when no physical dataset is provided.
"""

from pathlib import Path
from typing import Optional, Tuple, Union, Any
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except (ImportError, Exception):
    o3d = None
    HAS_OPEN3D = False


def numpy_to_o3d(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None
) -> Any:
    """
    Convert an (N, 3) NumPy array to an Open3D PointCloud object.
    """
    if not HAS_OPEN3D or o3d is None:
        raise RuntimeError("Open3D is not available in the current environment.")

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected points array of shape (N, 3) or (N, >=3), got {points.shape}")

    pcd = o3d.geometry.PointCloud()
    xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if colors is not None:
        if colors.shape[0] != points.shape[0] or colors.shape[1] != 3:
            raise ValueError(f"Colors must match points count (N, 3), got {colors.shape}")
        # Normalize to [0, 1] if given in [0, 255]
        if np.nanmax(colors) > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors, dtype=np.float64))

    return pcd


def o3d_to_numpy(pcd: Any) -> np.ndarray:
    """
    Extract (N, 3) coordinates from an Open3D PointCloud as a NumPy array.
    """
    if not HAS_OPEN3D or o3d is None:
        raise RuntimeError("Open3D is not available in the current environment.")
    return np.asarray(pcd.points)


def _read_ply_fallback(path: Path) -> np.ndarray:
    """
    Pure-Python / NumPy fallback reader for PLY point clouds (ASCII and binary little-endian).
    Ensures .ply files can be parsed even when Open3D is unavailable.
    """
    with open(str(path), "rb") as f:
        header_lines = []
        while True:
            raw_line = f.readline()
            if not raw_line:
                break
            line = raw_line.decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line == "end_header":
                break

    is_binary = any("binary_little_endian" in l for l in header_lines)
    is_ascii = any("format ascii" in l for l in header_lines)
    vertex_count = 0
    properties = []
    in_vertex = False

    for l in header_lines:
        parts = l.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            vertex_count = int(parts[2])
            in_vertex = True
        elif parts and parts[0] == "element" and parts[1] != "vertex":
            in_vertex = False
        elif in_vertex and len(parts) >= 3 and parts[0] == "property":
            properties.append((parts[1], parts[2]))

    if vertex_count == 0:
        raise ValueError(f"No vertices found in PLY file {path.name}")

    if is_ascii:
        with open(str(path), "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        header_end_idx = 0
        for i, l in enumerate(lines):
            if l.strip() == "end_header":
                header_end_idx = i + 1
                break
        data_lines = lines[header_end_idx:header_end_idx + vertex_count]
        points = np.array([[float(v) for v in dl.split()[:3]] for dl in data_lines], dtype=np.float64)
        return points

    elif is_binary:
        type_map = {
            "float": np.float32, "float32": np.float32,
            "double": np.float64, "float64": np.float64,
            "uchar": np.uint8, "uint8": np.uint8,
            "int": np.int32, "int32": np.int32
        }
        with open(str(path), "rb") as f:
            content = f.read()

        end_hdr = b"end_header\n"
        idx = content.find(end_hdr)
        if idx != -1:
            raw_payload = content[idx + len(end_hdr):]
        else:
            end_hdr_crlf = b"end_header\r\n"
            idx = content.find(end_hdr_crlf)
            raw_payload = content[idx + len(end_hdr_crlf):]

        prop_types = [p[0] for p in properties]
        # Fast path: only x, y, z
        if len(properties) >= 3 and all(p[1] in ["x", "y", "z"] for p in properties[:3]):
            dtype = np.float64 if "double" in prop_types[0] or "float64" in prop_types[0] else np.float32
            if len(properties) == 3:
                return np.frombuffer(raw_payload, dtype=dtype, count=vertex_count * 3).reshape(vertex_count, 3).astype(np.float64)
            else:
                # Vertex has extra fields; use structured dtype
                dt = np.dtype([(p[1], type_map.get(p[0], np.float32)) for p in properties])
                rec = np.frombuffer(raw_payload, dtype=dt, count=vertex_count)
                return np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64)

    raise ValueError(f"Unsupported PLY encoding in {path.name}")


def load_point_cloud(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load a point cloud from disk and return its points as an (N, 3) NumPy array.

    Supported formats:
    - .ply: Stanford polygon point cloud
    - .pcd: Point Cloud Data format
    - .bin: Raw binary float32 format (e.g., KITTI LiDAR: [x, y, z, intensity] or [x, y, z])
    - .npy: Saved NumPy array
    - .xyz, .txt, .csv: Delimited coordinate text files

    Args:
        file_path: Path to the point cloud file.

    Returns:
        np.ndarray: Array of shape (N, 3) with float64 coordinates.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file format is unsupported or point cloud has invalid structure.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {path.resolve()}")

    suffix = path.suffix.lower()

    if suffix in [".ply", ".pcd"]:
        if HAS_OPEN3D and o3d is not None:
            try:
                pcd = o3d.io.read_point_cloud(str(path))
                if not pcd.is_empty():
                    return np.asarray(pcd.points, dtype=np.float64)
            except Exception:
                pass

        if suffix == ".ply":
            return _read_ply_fallback(path)
        raise ValueError(
            f"Loading '{path.name}' requires Open3D which is not available in this environment."
        )

    elif suffix == ".bin":
        # KITTI and standard LiDAR binary files store consecutive float32 values
        raw_data = np.fromfile(str(path), dtype=np.float32)
        if raw_data.size == 0:
            raise ValueError(f"Binary file at {path.name} is empty.")
        if raw_data.size % 4 == 0:
            # Format: [x, y, z, intensity]
            points = raw_data.reshape(-1, 4)[:, :3].astype(np.float64)
        elif raw_data.size % 3 == 0:
            # Format: [x, y, z]
            points = raw_data.reshape(-1, 3).astype(np.float64)
        else:
            raise ValueError(
                f"Binary point cloud size ({raw_data.size} floats) is neither divisible by 4 nor 3."
            )

    elif suffix == ".npy":
        points = np.load(str(path))
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected .npy array with shape (N, >=3), got {points.shape}")
        points = points[:, :3].astype(np.float64)

    elif suffix in [".xyz", ".txt", ".csv"]:
        delimiter = "," if suffix == ".csv" else None
        points = np.loadtxt(str(path), delimiter=delimiter)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] < 3:
            raise ValueError(f"Text point file requires at least 3 columns (x, y, z), got {points.shape[1]}")
        points = points[:, :3].astype(np.float64)

    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. Supported: .ply, .pcd, .bin, .npy, .xyz, .txt, .csv"
        )

    return points


def save_point_cloud(
    file_path: Union[str, Path],
    points: np.ndarray,
    colors: Optional[np.ndarray] = None
) -> bool:
    """
    Save an (N, 3) point cloud to disk.

    Supports: .ply, .pcd, .bin, .npy
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix in [".ply", ".pcd"]:
        if HAS_OPEN3D and o3d is not None:
            pcd = numpy_to_o3d(points, colors=colors)
            return o3d.io.write_point_cloud(str(path), pcd)
        elif suffix == ".ply":
            # Pure-Python ASCII PLY writer fallback
            with open(str(path), "w", encoding="utf-8") as f:
                f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
                f.write("property double x\nproperty double y\nproperty double z\nend_header\n")
                for pt in points:
                    f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
            return True
        else:
            raise ValueError(f"Saving format '{suffix}' requires Open3D which is not available.")

    elif suffix == ".bin":
        # Save as float32 [x, y, z, 0.0] to mirror KITTI format
        xyz = points[:, :3].astype(np.float32)
        intensity = np.zeros((len(xyz), 1), dtype=np.float32)
        xyzi = np.hstack([xyz, intensity])
        xyzi.tofile(str(path))
        return True

    elif suffix == ".npy":
        np.save(str(path), points[:, :3].astype(np.float64))
        return True

    else:
        raise ValueError(f"Unsupported save format '{suffix}'. Supported: .ply, .pcd, .bin, .npy")


def generate_synthetic_lidar_scene(
    num_ground_points: int = 4000,
    noise_std: float = 0.02,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Create a synthetic LiDAR environment containing:
    1. Ground plane (-15m to +15m X/Y) with slight slope and sensor Gaussian noise.
    2. Vehicle obstacle (box structure).
    3. Pedestrian / pole obstacle (vertical cylinder structure).
    4. Wall / barrier obstacle (rectangular obstacle block).

    This provides a deterministic, realistic LiDAR scene for development and verification
    when no physical LiDAR hardware or dataset is connected.

    Returns:
        np.ndarray: (N, 3) array of synthetic point coordinates.
    """
    rng = np.random.default_rng(seed)

    # 1. Ground Plane: Flat surface near z = 0 with small sensor noise
    gx = rng.uniform(-15.0, 15.0, size=num_ground_points)
    gy = rng.uniform(-15.0, 15.0, size=num_ground_points)
    gz = rng.normal(0.0, noise_std, size=num_ground_points)
    ground_points = np.column_stack([gx, gy, gz])

    # 2. Vehicle Obstacle (box at X: 4 to 8, Y: 2 to 4, Z: 0 to 1.5)
    vx = rng.uniform(4.0, 8.0, size=800)
    vy = rng.uniform(2.0, 4.0, size=800)
    vz = rng.uniform(0.0, 1.5, size=800)
    vehicle_points = np.column_stack([vx, vy, vz])

    # 3. Pedestrian / Pole (cylinder at X: -3, Y: 5, radius: 0.3, Z: 0 to 1.8)
    theta = rng.uniform(0, 2 * np.pi, size=300)
    r = rng.uniform(0.0, 0.3, size=300)
    px = -3.0 + r * np.cos(theta)
    py = 5.0 + r * np.sin(theta)
    pz = rng.uniform(0.0, 1.8, size=300)
    pedestrian_points = np.column_stack([px, py, pz])

    # 4. Barrier / Wall (at X: -10 to -9, Y: -6 to 6, Z: 0 to 2.0)
    wx = rng.uniform(-10.0, -9.0, size=600)
    wy = rng.uniform(-6.0, 6.0, size=600)
    wz = rng.uniform(0.0, 2.0, size=600)
    wall_points = np.column_stack([wx, wy, wz])

    all_points = np.vstack([ground_points, vehicle_points, pedestrian_points, wall_points])
    return all_points


def get_or_create_fallback_pointcloud(
    output_path: Union[str, Path] = "data/sample_data/synthetic_scene.ply"
) -> np.ndarray:
    """
    Retrieve the fallback synthetic point cloud. If it exists on disk, load it;
    otherwise generate the deterministic scene and attempt to cache it to disk.
    Guaranteed to return valid (N, 3) point data without crashing.
    """
    path = Path(output_path)
    if path.exists():
        try:
            return load_point_cloud(path)
        except Exception as e:
            print(f"[Warning] Could not load fallback point cloud from {path}: {e}")

    points = generate_synthetic_lidar_scene()
    try:
        save_point_cloud(path, points)
    except Exception as e:
        print(f"[Warning] Could not cache fallback point cloud to {path}: {e}")
    return points
