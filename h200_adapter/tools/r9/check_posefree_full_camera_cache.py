#!/usr/bin/env python3
"""Validate a global metric pose-free cache for Full Waymo training.

The H200 Full-Waymo loader samples arbitrary scenes and arbitrary 20-frame
windows.  A cache containing one old single-scene window is therefore not
enough.  This checker requires one global_metric NPZ per train scene, with all
frames and the three cameras used by the R9 configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CAMERAS = ("1", "0", "2")
METRIC_SCALE_SOURCES = {
    # Current overlap aligner label.
    "moge2_gca_plus_overlap_camera_se3",
    # Older pose-free caches made before the label was made explicit.
    "moge2_rgb_correspondence_global_scale",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    return parser.parse_args()


def annotation_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def main() -> None:
    args = parse_args()
    contract_path = args.cache_root / ".posefree_contract.json"
    if not contract_path.is_file():
        # The training launcher points at global_aligned, while the producer
        # writes the contract at the cache root shared by all scenes.
        contract_path = args.cache_root.parent / ".posefree_contract.json"
    if not contract_path.is_file():
        raise RuntimeError(f"missing cache contract: {contract_path}")
    contract = json.loads(contract_path.read_text())
    if contract.get("scene_count") != 798:
        raise RuntimeError("cache contract is not Full Waymo (798 scenes)")
    if contract.get("camera_input_protocol") != "all_rgb":
        raise RuntimeError("cache contract is not RGB-only camera inference")
    if contract.get("metric_method") != "Omega + MoGe-2 + GCA + overlap SE(3)":
        raise RuntimeError("unexpected pose-free metric method")
    lines = [line.strip() for line in args.annotation_file.read_text().splitlines() if line.strip()]
    if len(lines) != 798:
        raise RuntimeError(f"expected 798 Full-Waymo train scenes, found {len(lines)}")

    failures: list[str] = []
    checked = 0
    for line in lines:
        scene_json = json.loads(annotation_path(args.data_root, line).read_text())
        scene = str(scene_json["scene_name"])
        path = args.cache_root / scene / "omega_pose_override.npz"
        if not path.is_file():
            failures.append(f"{scene}: missing {path}")
            continue

        try:
            with np.load(path, allow_pickle=False) as payload:
                payload_scene = str(payload["scene_name"].item())
                coordinate_frame = str(payload["coordinate_frame"].item())
                metric_scale_source = str(payload["metric_scale_source"].item())
                frame_ids = payload["frame_ids"].astype(np.int64)
                camera_ids = payload["camera_ids"].astype(str)
                pose_key = "omega_camera_to_world_global_metric"
                poses = payload[pose_key].astype(np.float64)
                intrinsics = payload["predicted_intrinsics_ufo"].astype(np.float64)

            if payload_scene != scene:
                raise ValueError(f"scene_name={payload_scene!r}")
            if coordinate_frame != "global_metric":
                raise ValueError(f"coordinate_frame={coordinate_frame!r}, expected 'global_metric'")
            if metric_scale_source not in METRIC_SCALE_SOURCES:
                raise ValueError(f"unexpected metric scale source: {metric_scale_source!r}")
            expected = {(frame, camera) for frame in range(int(scene_json["num_timesteps"])) for camera in CAMERAS}
            actual = {(int(frame), str(camera)) for frame, camera in zip(frame_ids, camera_ids)}
            if actual != expected:
                missing = sorted(expected - actual)[:3]
                extra = sorted(actual - expected)[:3]
                raise ValueError(f"frame/camera coverage mismatch; missing={missing}, extra={extra}")
            if poses.shape != (len(frame_ids), 4, 4):
                raise ValueError(f"pose shape={poses.shape}")
            if intrinsics.shape != (len(frame_ids), 3, 3):
                raise ValueError(f"intrinsics shape={intrinsics.shape}")
            if not np.isfinite(poses).all() or not np.isfinite(intrinsics).all():
                raise ValueError("non-finite pose or intrinsics")
        except Exception as error:
            failures.append(f"{scene}: {error}")
            continue
        checked += 1

    print(f"POSEFREE_GLOBAL_CACHE_SCENES={checked}/{len(lines)}")
    if failures:
        print("Invalid/missing examples:")
        print("\n".join(failures[:20]))
        raise SystemExit(2)
    print("POSEFREE_GLOBAL_CACHE=PASS")


if __name__ == "__main__":
    main()
