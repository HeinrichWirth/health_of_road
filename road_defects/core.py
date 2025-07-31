"""Core geometry + clustering pipeline."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import open3d as o3d
import pandas as pd

from .io import load_bin_frame, load_gps_csv, lookup_gps

_LOG = logging.getLogger(__name__)


@dataclass
class Config:
    """Pipeline parameters."""

    points_dir: Path
    gps_csv: Path

    point_step: int = 48
    frames: int = 75
    icp_threshold: float = 1.0

    # Algorithm parameters
    z_bins: int = 40
    sample_ratio: float = 0.10
    eps_sample: float = 0.10
    min_pts_sample: int = 20
    sigma: float = 4.0
    eps_defect: float = 0.06
    min_pts_defect: int = 3


class RoadDefectDetector:
    """Detect pavement defects using unlabelled LiDAR + GPS."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.gps_df = load_gps_csv(cfg.gps_csv)

    def process(self) -> None:
        """Iterate over all frames and log chunks that contain defects."""
        bin_files = self._discover_frames()
        if not bin_files:
            _LOG.error("No .bin frames found in %s", self.cfg.points_dir)
            return

        step = self.cfg.frames
        for i in range(0, len(bin_files), step):
            merged_cloud = self._merge_frames(bin_files[i : i + step])
            defect_cloud = self._detect_defects(merged_cloud)
            
            _LOG.info("Processed chunk, found %d defect points", len(defect_cloud.points))

            if len(defect_cloud.points) == 0:
                continue
            last_frame_in_chunk = bin_files[min(i + step - 1, len(bin_files) - 1)]
            lat, lon, alt = lookup_gps(self.gps_df, last_frame_in_chunk)

            _LOG.info(
                "Defects at %.6f, %.6f (alt %.1f m)  —  %d points",
                lat,
                lon,
                alt,
                len(defect_cloud.points),
            )

    def _discover_frames(self) -> list[Path]:
        """Return sorted list of .bin frames."""
        pattern = re.compile(r"rowid_(\d+)")
        return sorted(
            self.cfg.points_dir.glob("*.bin"),
            key=lambda p: int(pattern.search(p.stem).group(1)),
        )

    def _merge_frames(
        self, frames: Sequence[Path]
    ) -> o3d.geometry.PointCloud:
        """Merge frames into a denser point cloud via ICP."""
        merged = load_bin_frame(frames[0], self.cfg.point_step)

        for frame_path in frames[1:]:
            frame_cloud = load_bin_frame(frame_path, self.cfg.point_step)
            self._icp_align(frame_cloud, merged)
            merged += frame_cloud

        return merged

    def _icp_align(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
    ) -> None:
        """Rigidly align source onto target."""
        identity = np.eye(4)
        reg = o3d.pipelines.registration.registration_icp(
            source,
            target,
            self.cfg.icp_threshold,
            identity,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        source.transform(reg.transformation)

    # defect detection pipeline
    def _detect_defects(
        self, cloud: o3d.geometry.PointCloud
    ) -> o3d.geometry.PointCloud:
        """Return point cloud containing only defect clusters."""
        road_points = self._extract_road_surface(cloud)
        return self._cluster_outliers(road_points)

    def _extract_road_surface(self, cloud: o3d.geometry.PointCloud) -> np.ndarray:
        """Flatten ground plane and slice densest Z-layer (road surface)."""
        plane_model, _ = cloud.segment_plane(
            distance_threshold=0.1,
            ransac_n=50,
            num_iterations=4_000,
        )
        normal = np.asarray(plane_model[:3], dtype=float)
        normal /= np.linalg.norm(normal)

        axis = np.cross(normal, [0.0, 0.0, 1.0])
        angle = float(np.arccos(np.clip(normal @ [0, 0, 1], -1.0, 1.0)))

        if np.linalg.norm(axis) > 1e-8:
            rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
            cloud.rotate(rotation, center=(0.0, 0.0, 0.0))

        points = np.asarray(cloud.points)
        z_values = points[:, 2]

        histogram, edges = np.histogram(z_values, bins=self.cfg.z_bins)
        densest_idx = int(np.argmax(histogram))
        lower, upper = edges[densest_idx], edges[densest_idx + 1]

        return points[(z_values >= lower) & (z_values <= upper)]

    def _cluster_outliers(self, road_points: np.ndarray) -> o3d.geometry.PointCloud:
        """Cluster road surface and return clusters > 4 sigma below mean Z."""
        sample_size = max(1, int(len(road_points) * self.cfg.sample_ratio))
        sample_idx = np.random.choice(len(road_points), sample_size, replace=False)
        sample_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(road_points[sample_idx])
        )

        sample_labels = np.array(
            sample_cloud.cluster_dbscan(
                eps=self.cfg.eps_sample,
                min_points=self.cfg.min_pts_sample,
                print_progress=False,
            )
        )

        # propagate labels to full set via 1-NN lookup
        kd_tree = o3d.geometry.KDTreeFlann(sample_cloud)
        full_labels = np.array(
            [
                sample_labels[kd_tree.search_knn_vector_3d(p, 1)[1][0]]
                for p in road_points
            ]
        ) + 1

        counts = np.bincount(full_labels)
        if len(counts) <= 1:
            return o3d.geometry.PointCloud()

        dominant_label = int(np.argmax(counts[1:])) + 1
        surface_pts = road_points[full_labels == dominant_label]

        z_surface = surface_pts[:, 2]
        threshold = float(z_surface.mean() - self.cfg.sigma * z_surface.std())
        outliers = surface_pts[z_surface < threshold]

        outlier_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(outliers)
        )
        labels_defect = np.array(
            outlier_cloud.cluster_dbscan(
                eps=self.cfg.eps_defect,
                min_points=self.cfg.min_pts_defect,
                print_progress=False,
            )
        )

        valid_idx = np.where(labels_defect != -1)[0]
        return outlier_cloud.select_by_index(list(valid_idx))
