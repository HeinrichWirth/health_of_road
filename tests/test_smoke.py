"""Minimal smoke-test - just checks that package imports and config instantiates."""

from pathlib import Path

from road_defects import Config, RoadDefectDetector


def test_instantiation(tmp_path: Path) -> None:
    csv_path = tmp_path / "dummy.csv"
    csv_path.write_text("points_file_path,latitude,longitude,altitude\n")

    cfg = Config(points_dir=tmp_path, gps_csv=csv_path)
    detector = RoadDefectDetector(cfg)

    assert detector.cfg.frames == 75