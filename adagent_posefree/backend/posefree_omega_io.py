"""Lossless, atomic raw storage and cached image geometry for Omega."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import shutil
import tempfile
import time

import numpy as np


def write_npz(path, payload, compressed=False):
    started = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            save = np.savez_compressed if compressed else np.savez
            save(f, **payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"write_s": time.perf_counter() - started, "raw_bytes": path.stat().st_size}


class RawWriter:
    """One outstanding CPU write; failure is raised before accepting another.

    Caller may prepare one next window during the write. No unbounded queue
    of dense arrays; completed paths become visible only after ZIP closure.
    """
    def __init__(self, compressed=False):
        self.compressed = compressed
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.metadata = None

    def submit(self, path, payload, metadata):
        if self.future is not None:
            raise RuntimeError("collect the previous raw write before submitting")
        self.metadata = dict(metadata)
        self.future = self.pool.submit(write_npz, path, payload, self.compressed)

    def collect(self):
        if self.future is None:
            return None
        started = time.perf_counter()
        result = self.future.result()
        result.update(self.metadata)
        result["wait_writer_s"] = time.perf_counter() - started
        self.future = self.metadata = None
        return result

    def close(self):
        self.pool.shutdown(wait=True)


def require_disk_budget(root, payload_bytes, remaining):
    # One scene/batch is retained by the existing driver, then cleaned up.
    needed = int(payload_bytes * remaining * 1.10) + 1024**3
    free = shutil.disk_usage(root).free
    if free < needed:
        raise RuntimeError(
            f"Uncompressed raw cache needs about {needed / 1024**3:.1f} GiB free "
            f"for this batch; found {free / 1024**3:.1f} GiB. "
            "Use UFO_OMEGA_RAW_COMPRESSION=deflate or a smaller UFO_POSEFREE_WINDOW_BATCH_SIZE."
        )
    return needed


class IntrinsicsGeometryCache:
    def __init__(self, Image, balanced_shape):
        self.Image = Image
        self.balanced_shape = balanced_shape
        self.cache = {}

    def geometry(self, path):
        if path in self.cache:
            return self.cache[path]
        with self.Image.open(path) as image:
            original_width, original_height = image.size
        aspect = original_height / max(original_width, 1)
        left = top = 0
        width, height = original_width, original_height
        if aspect < 0.5:
            width = min(original_width, max(1, int(round(original_height / 0.5))))
            left = max((original_width - width) // 2, 0)
        elif aspect > 2.0:
            height = min(original_height, max(1, int(round(original_width * 2.0))))
            top = max((original_height - height) // 2, 0)
        resized_height, resized_width = self.balanced_shape(height / max(width, 1), 512, 16)
        value = dict(original_width=original_width, original_height=original_height,
                     crop_left=left, crop_top=top, crop_width=width, crop_height=height,
                     resized_width=resized_width, resized_height=resized_height)
        self.cache[path] = value
        return value

    def transform(self, intrinsics, paths, ufo_image_size):
        geometries = [self.geometry(path) for path in paths]
        padded_width = max(g["resized_width"] for g in geometries)
        padded_height = max(g["resized_height"] for g in geometries)
        ufo_height, ufo_width = ufo_image_size
        result = np.asarray(intrinsics, dtype=np.float64).copy()
        for i, g in enumerate(geometries):
            pad_left = (padded_width - g["resized_width"]) // 2
            pad_top = (padded_height - g["resized_height"]) // 2
            omega_scale_x = g["resized_width"] / g["crop_width"]
            omega_scale_y = g["resized_height"] / g["crop_height"]
            ufo_scale_x = ufo_width / g["original_width"]
            ufo_scale_y = ufo_height / g["original_height"]
            result[i, 0, 0] = intrinsics[i, 0, 0] / omega_scale_x * ufo_scale_x
            result[i, 1, 1] = intrinsics[i, 1, 1] / omega_scale_y * ufo_scale_y
            result[i, 0, 2] = ((intrinsics[i, 0, 2] - pad_left) / omega_scale_x + g["crop_left"]) * ufo_scale_x
            result[i, 1, 2] = ((intrinsics[i, 1, 2] - pad_top) / omega_scale_y + g["crop_top"]) * ufo_scale_y
        return result
