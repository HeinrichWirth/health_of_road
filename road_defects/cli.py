"""Tiny CLI wrapper - keeps main package free of argparse noise."""

import argparse
import logging
from pathlib import Path

from .core import Config, RoadDefectDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Road-surface defect detection"
    )
    parser.add_argument(
        "--points",
        type=Path,
        required=True,
        help="Directory with .bin LiDAR frames",
    )
    parser.add_argument(
        "--gps",
        type=Path,
        required=True,
        help="CSV with GPS metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    detector = RoadDefectDetector(
        Config(points_dir=args.points, gps_csv=args.gps)
    )
    detector.process()


if __name__ == "__main__":
    main()
