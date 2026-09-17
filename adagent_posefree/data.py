"""Explicit RGB-only input contract, independent of UFO annotations."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def safe_name(value):
    if not isinstance(value, str) or not value or value in (".", "..") or any(c in value for c in ("/", "\\", "\0", "\n", "\r")):
        raise ValueError(f"invalid scene/camera name: {value!r}")
    return value


def load_manifest(path, check_images=True):
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or not payload.get("scenes"):
        raise ValueError("manifest requires schema_version=1 and a nonempty scenes list")
    scenes, names = [], set()
    for original in payload["scenes"]:
        name = safe_name(original["scene_name"])
        if name in names:
            raise ValueError(f"duplicate scene: {name}")
        names.add(name)
        cameras = original["camera_ids"]
        if not cameras or len(cameras) != len(set(cameras)):
            raise ValueError(f"invalid camera_ids: {name}")
        cameras = [safe_name(c) for c in cameras]
        reference = original.get("reference_camera", cameras[0])
        if reference not in cameras:
            raise ValueError(f"reference camera missing in {name}")
        fps = float(original["fps"])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"invalid fps: {name}")
        basis = np.asarray(original.get("opencv_to_dataset", np.eye(4).tolist()), dtype=float)
        if (basis.shape != (4, 4) or not np.isfinite(basis).all()
                or not np.allclose(basis[3], [0, 0, 0, 1])
                or not np.allclose(basis[:3, 3], 0)
                or not np.allclose(basis[:3, :3].T @ basis[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(basis[:3, :3]), 1)):
            raise ValueError("opencv_to_dataset must be a proper, zero-translation axis rotation")
        frames, previous_id, previous_time = [], -1, -float("inf")
        for frame in original["frames"]:
            fid = frame["frame_id"]
            timestamp = float(frame.get("timestamp", fid / fps))
            if type(fid) is not int or fid <= previous_id or not math.isfinite(timestamp) or timestamp <= previous_time:
                raise ValueError(f"frames must have increasing nonnegative IDs and timestamps: {name}")
            if set(frame["images"]) != set(cameras):
                raise ValueError(f"missing/extra camera at {name} frame {fid}")
            images = {}
            for camera in cameras:
                image = Path(frame["images"][camera])
                image = (path.parent / image).resolve() if not image.is_absolute() else image.resolve()
                if check_images and not image.is_file():
                    raise FileNotFoundError(image)
                images[camera] = str(image)
            frames.append(dict(frame_id=fid, timestamp=timestamp, images=images))
            previous_id, previous_time = fid, timestamp
        if len(frames) * len(cameras) < 2 or not frames:
            raise ValueError(f"need at least two images: {name}")
        scenes.append(dict(scene_name=name, camera_ids=cameras, reference_camera=reference,
                           fps=fps, opencv_to_dataset=basis.tolist(), frames=frames))
    return scenes


def window_starts(length, window, stride):
    if window < 2 or stride < 1 or stride >= window:
        raise ValueError("require window_frames >= 2 and 1 <= window_stride < window_frames")
    last = max(0, length - window)
    return sorted(set([*range(0, last + 1, stride), last]))


def window_manifest(scene, start, config):
    entries = []
    for offset, frame in enumerate(scene["frames"][start:start + config["window_frames"]]):
        for camera in scene["camera_ids"]:
            entries.append(dict(frame_id=frame["frame_id"], camera_id=camera,
                                role="context" if offset % 5 == 0 else "target",
                                chunk_index=offset // 5, path=frame["images"][camera]))
    return dict(schema_version=2, scene_name=scene["scene_name"], start_index=start,
                camera_ids=scene["camera_ids"], reference_camera=scene["reference_camera"],
                ufo_image_size=config["output_image_size"], opencv_to_dataset=scene["opencv_to_dataset"],
                pose_contract={"sensor_inputs": ["rgb"], "camera_input_protocol": "all_rgb"}, images=entries)


def index_folders(root, fps, reference=None):
    """root/scene/camera/000001.jpg; numeric stems define synchronized frames."""
    root = Path(root).resolve()
    scenes = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        cameras, maps = [], {}
        for camera_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
            images = {}
            for path in sorted(camera_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                if not path.stem.isdigit() or int(path.stem) in images:
                    raise ValueError(f"image stems must be distinct integer frame IDs: {path}")
                images[int(path.stem)] = str(path.resolve())
            if images:
                cameras.append(camera_dir.name)
                maps[camera_dir.name] = images
        if not cameras:
            raise ValueError(f"no camera image folders in {directory}")
        ids = sorted(maps[cameras[0]])
        if any(set(maps[c]) != set(ids) for c in cameras):
            raise ValueError(f"camera frame IDs are not synchronized: {directory}")
        ref = reference if reference is not None else cameras[0]
        if ref not in cameras:
            raise ValueError(f"reference camera {ref} missing in {directory}")
        scenes.append(dict(scene_name=directory.name, fps=fps, camera_ids=cameras, reference_camera=ref,
                           frames=[dict(frame_id=i, timestamp=i/fps,
                                        images={c: maps[c][i] for c in cameras}) for i in ids]))
    return dict(schema_version=1, scenes=scenes)


def index_waymo(data_root, annotation, cameras=("1", "0", "2")):
    """Read only RGB paths and timing; accept any number of scenes."""
    root = Path(data_root).resolve()
    scenes = []
    for line in Path(annotation).read_text().splitlines():
        if not line.strip():
            continue
        path = Path(line.strip())
        s = json.loads((path if path.is_absolute() else root / path).read_text())
        if s["dataset"] != "waymo":
            raise ValueError("index-waymo only accepts Waymo; use index or an explicit manifest for other data")
        frames = []
        for i in range(int(s["num_timesteps"])):
            images = {c: str(root / "datasets/waymo" / s["relative_image_path"][c][i].replace("images", "images_4"))
                      for c in cameras}
            frames.append(dict(frame_id=i, timestamp=i / float(s["fps"]), images=images))
        scenes.append(dict(scene_name=s["scene_name"], fps=s["fps"], camera_ids=list(cameras),
                           reference_camera="0", frames=frames,
                           opencv_to_dataset=[[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]]))
    return dict(schema_version=1, scenes=scenes)
