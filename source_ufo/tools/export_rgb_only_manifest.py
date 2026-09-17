#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ufo.dataset.constants import DATASET_DICT, DATASETS
from ufo.paper_contract import split_context_supervision


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--annotation-file", type=Path, required=True)
    p.add_argument("--scene-index", type=int, required=True)
    p.add_argument("--start-index", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def image_path(data_root, dataset_name, relative_path):
    if dataset_name in ("waymo", "nuscenes", "argoverse2", "argoverse"):
        relative_path = relative_path.replace("images", "images_4")
        relative_path = relative_path.replace("sweeps", "sweeps_4")
        relative_path = relative_path.replace("samples", "samples_4")
    return data_root / "datasets" / dataset_name / relative_path


def main():
    args = parse_args()
    cfg = json.loads(args.config.read_text())

    lines = [
        x.strip()
        for x in args.annotation_file.read_text().splitlines()
        if x.strip()
    ]
    annotation_path = Path(lines[args.scene_index])
    if not annotation_path.is_absolute():
        annotation_path = args.data_root / annotation_path

    scene = json.loads(annotation_path.read_text())

    fps = scene["fps"]
    frames_per_chunk = int(cfg["timespan"] * fps)
    total_frames = frames_per_chunk * cfg["num_target_chunks"]

    start = args.start_index
    end = start + total_frames

    if start < 0 or end > scene["num_timesteps"]:
        raise ValueError(
            f"invalid window [{start},{end}) for "
            f"{scene['num_timesteps']} frames"
        )

    dataset_name = scene["dataset"]
    cameras = DATASET_DICT[dataset_name]["camera_list"][
        cfg["num_max_cameras"]
    ]

    entries = []
    seen = set()

    for chunk_idx in range(cfg["num_target_chunks"]):
        chunk_start = start + chunk_idx * frames_per_chunk
        chunk_end = chunk_start + frames_per_chunk
        protocol = split_context_supervision(chunk_start, chunk_end)

        for role, frame_ids in (
            ("context", protocol.context),
            ("target", protocol.supervision),
        ):
            for frame_id in frame_ids:
                for camera_id in cameras:
                    key = (int(frame_id), str(camera_id))
                    if key in seen:
                        raise RuntimeError(f"duplicate RGB entry: {key}")
                    seen.add(key)

                    rel = scene["relative_image_path"][camera_id][frame_id]

                    entries.append({
                        "frame_id": int(frame_id),
                        "camera_id": str(camera_id),
                        "role": role,
                        "chunk_index": chunk_idx,
                        "path": str(
                            image_path(
                                args.data_root,
                                dataset_name,
                                rel,
                            ).resolve()
                        ),
                    })

    # This is a coordinate-convention constant, not camera calibration.
    opencv_to_dataset = np.asarray(
        DATASETS[dataset_name]["opencv2dataset"],
        dtype=np.float64,
    )

    manifest = {
        "schema_version": 2,
        "pose_contract": {
            "name": "rgb_only_camera_prediction_v1",
            "sensor_inputs": ["rgb"],
            "camera_source": "VGGT-Omega",
            "metric_source": "MoGe-2 + GCA scale",
            "forbidden_sources": [
                "gt_camera_pose",
                "gt_intrinsics",
                "camera_to_ego",
                "ego_pose",
                "lidar",
                "gt_depth",
            ],
        },
        "scene_index": args.scene_index,
        "scene_id": scene["scene_id"],
        "scene_name": scene["scene_name"],
        "dataset": dataset_name,
        "start_index": start,
        "camera_ids": list(cameras),
        "ufo_image_size": cfg["input_size"],
        "opencv_to_dataset": opencv_to_dataset.tolist(),
        "images": entries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({
        "output": str(args.output),
        "start": start,
        "end": end,
        "images": len(entries),
        "gt_geometry": False,
    }, indent=2))


if __name__ == "__main__":
    main()
