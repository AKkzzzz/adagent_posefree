"""Input caching and metric conversion preserved from the R9 snapshot."""
import concurrent.futures
import os
import json
from pathlib import Path
import numpy as np
from .posefree_omega_io import write_npz

def homo(extri: np.ndarray) -> np.ndarray:
    out = np.tile(np.eye(4), (len(extri), 1, 1))
    out[:, :3, :4] = extri
    return out

def prepare_rgb_cache(paths, load_rgb, crop, target_shape, to_tensor, image_cls):
    """Decode/resize each unique scene image once on CPU."""
    def one(path):
        image = crop(load_rgb(path))
        width, height = image.size
        target_h, target_w = target_shape(
            height / max(width, 1), 512, 16
        )
        image = image.resize((target_w, target_h), image_cls.Resampling.BICUBIC)
        return path, to_tensor(image)

    workers = min(8, max(1, os.cpu_count() or 1), len(paths))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, paths))

def stack_cached_rgb(paths, cache, pad_images, torch):
    images = [cache[path] for path in paths]
    shapes = {(int(image.shape[1]), int(image.shape[2])) for image in images}
    if len(shapes) > 1:
        images = pad_images(images, shapes)
    return torch.stack(images)

def metric_output(manifest: dict, raw_path: Path, scale_path: Path, out: Path) -> None:
    report = json.loads(scale_path.read_text())
    scale = float(report["global_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid scale: {scale}")
    with np.load(raw_path, allow_pickle=False) as x:
        frame_ids = x["frame_ids"].astype(np.int32)
        camera_ids = x["camera_ids"].astype(str)
        roles = x["roles"].astype(str)
        raw_c2w = x["omega_c2w_raw"].astype(np.float64)
        raw_w2c = x["omega_w2c_raw"].astype(np.float64)
        k = x["predicted_intrinsics_ufo"].astype(np.float64)
    first_frame = int(frame_ids.min())
    reference = manifest["reference_camera"]
    idx = np.flatnonzero((frame_ids == first_frame) & (camera_ids == reference))
    if len(idx) != 1:
        raise ValueError(f"expected exactly one reference camera at frame {first_frame}")
    local_c2w = np.einsum("ij,njk->nik", np.linalg.inv(raw_c2w[int(idx[0])]), raw_c2w)
    local_c2w[:, :3, 3] *= scale
    local_w2c = np.linalg.inv(local_c2w)
    opencv_to_dataset = np.asarray(manifest["opencv_to_dataset"], dtype=np.float64)
    local_native = local_c2w @ np.linalg.inv(opencv_to_dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_npz(out, dict(scene_name=np.asarray(manifest["scene_name"]), scope=np.asarray("all"),
        coordinate_frame=np.asarray("rig_local_metric"), metric_scale_source=np.asarray("moge2_rgb_correspondence_global_scale"),
        world_gauge=np.asarray("first_timestamp_reference_camera"), frame_ids=frame_ids, camera_ids=camera_ids, roles=roles,
        omega_w2c_raw=raw_w2c.astype(np.float32), omega_c2w_raw=raw_c2w.astype(np.float32),
        omega_w2c_rig_local=local_w2c.astype(np.float32), omega_c2w_rig_local=local_c2w.astype(np.float32),
        omega_camera_to_world_rig_local=local_native.astype(np.float32), predicted_intrinsics_ufo=k.astype(np.float32),
        rgb_metric_scale=np.asarray(scale), gauge_frame_id=np.asarray(first_frame), gauge_camera_id=np.asarray(reference)), compressed=True)
