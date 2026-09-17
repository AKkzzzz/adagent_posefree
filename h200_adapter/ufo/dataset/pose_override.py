"""Load per-scene camera poses exported by VGGT-Omega."""

from pathlib import Path

import numpy as np


class PoseOverrideStore:
    def __init__(self, root):
        self.root = Path(root)
        self._scene_name = None
        self._poses = None
        self._intrinsics = None
        self._coordinate_frame = None

    def _load(self, scene_name):
        path = self.root / scene_name / "omega_pose_override.npz"
        if not path.is_file():
            raise FileNotFoundError(f"pose override not found: {path}")
        with np.load(path, allow_pickle=False) as payload:
            payload_scene = str(payload["scene_name"].item())
            if payload_scene != scene_name:
                raise ValueError(
                    f"pose override scene mismatch: {payload_scene!r} != {scene_name!r}"
                )
            frame_ids = payload["frame_ids"].astype(np.int64)
            camera_ids = payload["camera_ids"].astype(str)
            coordinate_frame = (
                str(payload["coordinate_frame"].item())
                if "coordinate_frame" in payload else "gt_sim3_aligned"
            )
            if coordinate_frame == "rig_local_metric":
                pose_key = "omega_camera_to_world_rig_local"
            elif coordinate_frame == "global_metric":
                pose_key = "omega_camera_to_world_global_metric"
            else:
                pose_key = "omega_camera_to_world_aligned"
            poses = payload[pose_key].astype(np.float64)
            intrinsics = (
                payload["predicted_intrinsics_ufo"].astype(np.float64)
                if "predicted_intrinsics_ufo" in payload else None
            )
        if poses.shape != (len(frame_ids), 4, 4):
            raise ValueError(f"invalid pose override shape {poses.shape} in {path}")
        if intrinsics is not None and intrinsics.shape != (len(frame_ids), 3, 3):
            raise ValueError(f"invalid intrinsics override shape {intrinsics.shape} in {path}")
        mapping = {}
        for frame_id, camera_id, pose in zip(frame_ids, camera_ids, poses):
            key = (int(frame_id), str(camera_id))
            if key in mapping:
                raise ValueError(f"duplicate pose override entry {key} in {path}")
            mapping[key] = pose
        self._scene_name = scene_name
        self._coordinate_frame = coordinate_frame
        self._poses = mapping
        self._intrinsics = None if intrinsics is None else {
            (int(frame_id), str(camera_id)): intrinsic
            for frame_id, camera_id, intrinsic in zip(frame_ids, camera_ids, intrinsics)
        }

    def coordinate_frame(self, scene_name):
        if self._scene_name != scene_name:
            self._load(scene_name)
        return self._coordinate_frame

    def frame_ids(self, scene_name):
        if self._scene_name != scene_name:
            self._load(scene_name)
        return tuple(sorted({frame_id for frame_id, _ in self._poses}))

    def get(self, scene_name, frame_id, camera_id):
        if self._scene_name != scene_name:
            self._load(scene_name)
        key = (int(frame_id), str(camera_id))
        try:
            return self._poses[key]
        except KeyError as error:
            raise KeyError(
                f"pose override for scene={scene_name} frame={frame_id} camera={camera_id} is missing"
            ) from error

    def get_intrinsics(self, scene_name, frame_id, camera_id):
        if self._scene_name != scene_name:
            self._load(scene_name)
        if self._intrinsics is None:
            raise KeyError(
                f"intrinsics override is absent for scene={scene_name}; regenerate the Omega NPZ"
            )
        key = (int(frame_id), str(camera_id))
        try:
            return self._intrinsics[key]
        except KeyError as error:
            raise KeyError(
                f"intrinsics override for scene={scene_name} frame={frame_id} "
                f"camera={camera_id} is missing"
            ) from error
