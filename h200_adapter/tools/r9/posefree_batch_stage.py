#!/usr/bin/env python3
"""Run one pose-free preparation stage over several windows per process.

The original preparation driver starts a fresh Python process (and reloads the
Omega/MoGe weights) for every 20-frame window.  This helper keeps one model
loaded while processing a bounded list of windows.  It deliberately keeps the
old per-window numerical operations and file schema.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path

import numpy as np


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("omega", "gca", "metric"), required=True)
    p.add_argument("--source-ufo-root", type=Path, required=True)
    p.add_argument("--omega-repo", type=Path)
    p.add_argument("--omega-checkpoint", type=Path)
    p.add_argument("--moge-repo", type=Path)
    p.add_argument("--moge-model", type=Path)
    p.add_argument("--manifest-list", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--scale-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    return p.parse_args()


def manifests(path: Path) -> list[Path]:
    return [Path(x.strip()) for x in path.read_text().splitlines() if x.strip()]


def raw_ok(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as x:
            return all(key in x for key in (
                "scene_name", "frame_ids", "camera_ids", "roles",
                "omega_w2c_raw", "omega_c2w_raw", "predicted_intrinsics_ufo",
                "omega_depth_raw", "omega_depth_conf_raw",
            ))
    except Exception:
        return False


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


def raw_output(manifest: dict, out: Path, model, load_images, transform_intrinsics, torch) -> None:
    entries = manifest["images"]
    paths = [e["path"] for e in entries]
    images = load_images(paths, image_resolution=512).to("cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(images)
        extri, intri = __import__("vggt_omega.utils.pose_enc", fromlist=["encoding_to_camera"]).encoding_to_camera(
            pred["pose_enc"], pred["images"].shape[-2:]
        )
        depth = pred["depth"][0].float().cpu().numpy()
        conf = pred["depth_conf"][0].float().cpu().numpy()

    w2c = homo(extri[0].float().cpu().numpy())
    c2w = np.linalg.inv(w2c)
    k = intri[0].float().cpu().numpy()
    k_ufo, _ = transform_intrinsics(k, paths, manifest["ufo_image_size"], image_resolution=512)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        scene_name=np.asarray(manifest["scene_name"]), scope=np.asarray("all"),
        frame_ids=np.asarray([e["frame_id"] for e in entries], dtype=np.int32),
        camera_ids=np.asarray([e["camera_id"] for e in entries]),
        roles=np.asarray([e["role"] for e in entries]),
        omega_w2c_raw=w2c.astype(np.float32), omega_c2w_raw=c2w.astype(np.float32),
        predicted_intrinsics_ufo=k_ufo.astype(np.float32),
        omega_depth_raw=depth.astype(np.float32), omega_depth_conf_raw=conf.astype(np.float32),
    )
    del pred, images, extri, intri


def gca_output(manifest: dict, raw_path: Path, out: Path, moge, transform_to_omega, top_n, percentile, torch, Image) -> None:
    from diagnose_gca_omega_scale import transform_to_omega as _unused  # noqa: F401
    with np.load(raw_path, allow_pickle=False) as x:
        depth = torch.from_numpy(x["omega_depth_raw"]).float()
        conf = torch.from_numpy(x["omega_depth_conf_raw"]).float()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if depth.ndim != 3 or conf.ndim != 3:
        raise RuntimeError(f"unexpected cached shapes: depth={depth.shape}, conf={conf.shape}")
    entries = manifest["images"]
    selected_scores = []
    for i in range(len(entries)):
        threshold = torch.quantile(conf[i].float(), percentile)
        selected_scores.append((conf[i][conf[i] > threshold].sum().item(), i))
    selected_scores.sort(reverse=True)
    selected = [i for _, i in selected_scores[: min(top_n, len(entries))]]
    all_ratios, rows = [], []
    oh, ow = depth.shape[-2:]
    for idx in selected:
        entry = entries[idx]
        rgb_np = np.asarray(Image.open(entry["path"]).convert("RGB")).copy()
        rgb = torch.from_numpy(rgb_np).float().permute(2, 0, 1).to("cuda") / 255.0
        with torch.inference_mode():
            result = moge.infer(rgb, resolution_level=9, use_fp16=True, apply_mask=False)
        metric_depth = result["depth"].float().cpu()
        metric_mask = result["mask"].cpu().bool()
        metric_depth = transform_to_omega(metric_depth, (oh, ow), is_mask=False)
        metric_mask = transform_to_omega(metric_mask, (oh, ow), is_mask=True)
        threshold = torch.quantile(conf[idx].float(), percentile)
        valid = ((conf[idx] > threshold) & metric_mask & torch.isfinite(metric_depth)
                 & torch.isfinite(depth[idx]) & (depth[idx] > 1e-4) & (metric_depth > 1e-4))
        n = int(valid.sum())
        if n < 100:
            continue
        ratios = (metric_depth[valid] / depth[idx][valid]).float()
        rows.append({"index": idx, "frame": int(entry["frame_id"]), "camera": str(entry["camera_id"]),
                     "valid_pixels": n, "median_scale": float(torch.median(ratios).item())})
        all_ratios.append(ratios)
        del result, rgb, metric_depth, metric_mask
    if not all_ratios:
        raise RuntimeError("GCA scale estimation failed")
    ratios = torch.cat(all_ratios)
    scale = torch.median(ratios).item()
    logs = torch.log(ratios)
    med_log = torch.median(logs)
    report = {"method": "gca_metric_scale_adapted_to_vggt_omega", "source_method": "GCA estimate_scale",
              "top_n": top_n, "relative_conf_percentile": percentile, "global_scale": float(scale),
              "global_log_mad": float(torch.median(torch.abs(logs - med_log)).item()),
              "selected": rows, "num_ratio_pixels": int(ratios.numel()), "uses_gt_geometry": False,
              "uses_gt_pose": False, "uses_gt_intrinsics": False, "uses_gt_depth": False,
              "uses_camera_to_ego": False, "omega_source": "cached_rgb_only_forward"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")


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
    idx = np.flatnonzero((frame_ids == first_frame) & (camera_ids == "0"))
    if len(idx) != 1:
        raise ValueError(f"expected exactly one front camera at frame {first_frame}")
    local_c2w = np.einsum("ij,njk->nik", np.linalg.inv(raw_c2w[int(idx[0])]), raw_c2w)
    local_c2w[:, :3, 3] *= scale
    local_w2c = np.linalg.inv(local_c2w)
    opencv_to_dataset = np.asarray(manifest["opencv_to_dataset"], dtype=np.float64)
    local_native = local_c2w @ np.linalg.inv(opencv_to_dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, scene_name=np.asarray(manifest["scene_name"]), scope=np.asarray("all"),
        coordinate_frame=np.asarray("rig_local_metric"), metric_scale_source=np.asarray("moge2_rgb_correspondence_global_scale"),
        world_gauge=np.asarray("first_timestamp_front_camera"), frame_ids=frame_ids, camera_ids=camera_ids, roles=roles,
        omega_w2c_raw=raw_w2c.astype(np.float32), omega_c2w_raw=raw_c2w.astype(np.float32),
        omega_w2c_rig_local=local_w2c.astype(np.float32), omega_c2w_rig_local=local_c2w.astype(np.float32),
        omega_camera_to_world_rig_local=local_native.astype(np.float32), predicted_intrinsics_ufo=k.astype(np.float32),
        rgb_metric_scale=np.asarray(scale), gauge_frame_id=np.asarray(first_frame), gauge_camera_id=np.asarray("0"))


def main() -> None:
    a = args()
    paths = manifests(a.manifest_list)
    if not paths:
        return
    sys.path.insert(0, str(a.source_ufo_root / "tools"))
    if a.omega_repo:
        sys.path.insert(0, str(a.omega_repo / "tools"))
    if a.stage == "omega":
        from posefree_omega_runtime import run_omega
        run_omega(a, paths, raw_ok, prepare_rgb_cache, stack_cached_rgb, homo)
        return
    if a.stage == "gca":
        from posefree_gca_runtime import run_gca
        run_gca(a, paths)
        return
    for path in paths:
        raw = a.raw_root / path.parent.name / (path.stem + ".npz")
        scale = a.scale_root / path.parent.name / (path.stem + ".json")
        out = a.output_root / path.parent.name / path.stem / "omega_pose_override.npz"
        if out.is_file():
            continue
        metric_output(json.loads(path.read_text()), raw, scale, out)
        print(f"[batch/metric] {path.name}", flush=True)


if __name__ == "__main__":
    main()
