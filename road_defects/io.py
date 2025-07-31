"""utilities: read LiDAR .bin frames and GPS CSV."""

from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
import pandas as pd


def load_bin_frame(path: Path, point_step: int = 48) -> o3d.geometry.PointCloud:
    """Read one raw .bin file and return an open3d.geometry.PointCloud."""
    buffer = np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(-1, point_step)

    x = np.frombuffer(buffer[:, 0:4].tobytes(), dtype=np.float32)
    y = np.frombuffer(buffer[:, 4:8].tobytes(), dtype=np.float32)
    z = np.frombuffer(buffer[:, 8:12].tobytes(), dtype=np.float32)

    cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.column_stack((x, y, z)))
    )

    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return cloud


def load_gps_csv(csv_path: Path) -> pd.DataFrame:
    """Load GPS metadata (columns: points_file_path, latitude, longitude, altitude)."""
    df = pd.read_csv(csv_path)
    df["points_file_path"] = df["points_file_path"].str.split("/").str[-1]
    return df


def lookup_gps(df: pd.DataFrame, bin_path: Path) -> Tuple[float, float, float]:
    """Return (lat, lon, alt) for given .bin frame."""
    fname = f"{bin_path.stem}.json"
    row = df[df["points_file_path"] == fname]

    if row.empty:
        return float("nan"), float("nan"), float("nan")

    record = row.iloc[0]
    return record.latitude, record.longitude, record.altitude
