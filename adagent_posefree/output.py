"""Validated, atomic camera exports and explicit image geometry."""
import json
import zipfile
from pathlib import Path

import numpy as np
from .backend.posefree_omega_io import write_npz
from .backend.posefree_gca_runtime import atomic_json
from .data import digest


def expected_keys(scene):
    return {(f["frame_id"], c) for f in scene["frames"] for c in scene["camera_ids"]}


def validate_arrays(payload, scene, pose_key="c2w", k_key="K"):
    f, c = payload["frame_ids"], payload["camera_ids"].astype(str)
    if f.ndim != 1 or c.shape != f.shape or f.dtype.kind not in "iu":
        raise ValueError("invalid frame/camera keys")
    actual = set(zip(f.tolist(), c.tolist()))
    if len(f) != len(actual) or actual != expected_keys(scene):
        raise ValueError("missing, extra or duplicate frame/camera records")
    p, k = payload[pose_key], payload[k_key]
    if p.shape != (len(f), 4, 4) or k.shape != (len(f), 3, 3):
        raise ValueError("invalid camera array shape")
    if not np.isfinite(p).all() or not np.isfinite(k).all() or (k[:, (0, 1), (0, 1)] <= 0).any():
        raise ValueError("nonfinite camera values or nonpositive focal length")
    if not np.allclose(p[:, 3], [0, 0, 0, 1], atol=1e-5) or not np.allclose(k[:, 2], [0, 0, 1], atol=1e-5):
        raise ValueError("invalid homogeneous camera matrix")
    rot = p[:, :3, :3]
    if not np.allclose(np.swapaxes(rot, 1, 2) @ rot, np.eye(3), atol=2e-4) or not np.allclose(np.linalg.det(rot), 1, atol=2e-4):
        raise ValueError("camera rotation is not in SO(3)")


def complete(root, scene, signature=None):
    folder = Path(root) / "scenes" / scene["scene_name"]
    try:
        done = json.loads((folder / "done.json").read_text())
        metadata = json.loads((folder / "metadata.json").read_text())
        if (done["scene_hash"] != digest(scene) or metadata["scene_name"] != scene["scene_name"]
                or (signature is not None and done["run_signature"] != signature)):
            return False
        with np.load(folder / "cameras.npz", allow_pickle=False) as x:
            validate_arrays(x, scene)
            if list(x["image_size"].tolist()) != metadata["image_size"]:
                return False
        with np.load(Path(root) / "global_aligned" / scene["scene_name"] / "omega_pose_override.npz", allow_pickle=False) as x:
            validate_arrays(x, scene, "omega_camera_to_world_global_metric", "predicted_intrinsics_ufo")
        return True
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        return False


def export_scene(root, scene, cameras, report, config, signature):
    root = Path(root)
    validate_arrays(cameras, scene)
    validate_arrays(cameras, scene, "camera_to_world_dataset")
    folder = root / "scenes" / scene["scene_name"]
    by_frame = {f["frame_id"]: f for f in scene["frames"]}
    image_paths = [by_frame[int(f)]["images"][str(c)] for f, c in zip(cameras["frame_ids"], cameras["camera_ids"])]
    timestamps = [by_frame[int(f)]["timestamp"] for f in cameras["frame_ids"]]
    output = dict(cameras, timestamps=np.asarray(timestamps), image_paths=np.asarray(image_paths),
                  image_size=np.asarray(config["output_image_size"], dtype=np.int32))
    write_npz(folder / "cameras.npz", output, compressed=True)
    world_gauge = f"initial_window_frame_{scene['frames'][0]['frame_id']}_camera_{scene['reference_camera']}"
    write_npz(root / "global_aligned" / scene["scene_name"] / "omega_pose_override.npz", dict(
        scene_name=np.asarray(scene["scene_name"]), scope=np.asarray("all"), coordinate_frame=np.asarray("global_metric"),
        metric_scale_source=np.asarray("moge2_gca_plus_overlap_camera_se3"), world_gauge=np.asarray(world_gauge),
        frame_ids=cameras["frame_ids"], camera_ids=cameras["camera_ids"],
        omega_c2w_global_metric=cameras["c2w"], omega_camera_to_world_global_metric=cameras["camera_to_world_dataset"],
        predicted_intrinsics_ufo=cameras["K"]), compressed=True)
    atomic_json(folder / "alignment_report.json", report)
    atomic_json(folder / "metadata.json", dict(schema_version=1, scene_name=scene["scene_name"],
        image_size=config["output_image_size"], image_size_order="height,width", camera_basis="OpenCV: right,down,forward",
        transform="c2w: camera to per-scene world; w2c=inverse(c2w)",
        image_transform="K refers to a direct resize of each original input RGB to image_size; no target crop",
        reference_camera=scene["reference_camera"], world_gauge=world_gauge,
        metric_scale_source="MoGe-2/GCA estimated metric scale; not GPS or measured ground truth",
        camera_input_protocol="all_rgb (offline, all supplied window images)",
        opencv_to_dataset=scene["opencv_to_dataset"], run_signature=signature))
    # Publish the completion marker last, after both output formats exist.
    atomic_json(folder / "done.json", dict(scene_hash=digest(scene), run_signature=signature))
    if not complete(root, scene, signature):
        (folder / "done.json").unlink(missing_ok=True)
        raise RuntimeError("final camera validation failed; intermediates retained")
