#!/usr/bin/env python3
"""Infer VGGT-Omega poses for a UFO manifest and align them to GT world."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import (
    _balanced_target_shape,
    load_and_preprocess_images,
)
from vggt_omega.utils.pose_enc import encoding_to_camera


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("context", "all"), required=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    return parser.parse_args()


def as_homogeneous(extrinsics):
    result = np.tile(np.eye(4, dtype=np.float64), (len(extrinsics), 1, 1))
    result[:, :3, :4] = extrinsics
    return result


def transform_intrinsics_to_ufo(
    intrinsics, image_paths, ufo_image_size, image_resolution=512, patch_size=16
):
    """Map K from Omega's crop/resize/pad tensor into UFO image pixels."""
    geometries = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            original_width, original_height = image.size
        aspect_ratio = original_height / max(original_width, 1)
        crop_left = crop_top = 0
        crop_width, crop_height = original_width, original_height
        if aspect_ratio < 0.5:
            crop_width = min(original_width, max(1, int(round(original_height / 0.5))))
            crop_left = max((original_width - crop_width) // 2, 0)
        elif aspect_ratio > 2.0:
            crop_height = min(original_height, max(1, int(round(original_width * 2.0))))
            crop_top = max((original_height - crop_height) // 2, 0)
        cropped_aspect_ratio = crop_height / max(crop_width, 1)
        resized_height, resized_width = _balanced_target_shape(
            cropped_aspect_ratio, image_resolution, patch_size
        )
        geometries.append({
            "original_width": original_width,
            "original_height": original_height,
            "crop_left": crop_left,
            "crop_top": crop_top,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "resized_width": resized_width,
            "resized_height": resized_height,
        })

    padded_width = max(item["resized_width"] for item in geometries)
    padded_height = max(item["resized_height"] for item in geometries)
    ufo_height, ufo_width = ufo_image_size
    transformed = np.asarray(intrinsics, dtype=np.float64).copy()
    for index, geometry in enumerate(geometries):
        pad_left = (padded_width - geometry["resized_width"]) // 2
        pad_top = (padded_height - geometry["resized_height"]) // 2
        omega_scale_x = geometry["resized_width"] / geometry["crop_width"]
        omega_scale_y = geometry["resized_height"] / geometry["crop_height"]
        ufo_scale_x = ufo_width / geometry["original_width"]
        ufo_scale_y = ufo_height / geometry["original_height"]
        transformed[index, 0, 0] = (
            intrinsics[index, 0, 0] / omega_scale_x * ufo_scale_x
        )
        transformed[index, 1, 1] = (
            intrinsics[index, 1, 1] / omega_scale_y * ufo_scale_y
        )
        transformed[index, 0, 2] = (
            (intrinsics[index, 0, 2] - pad_left) / omega_scale_x
            + geometry["crop_left"]
        ) * ufo_scale_x
        transformed[index, 1, 2] = (
            (intrinsics[index, 1, 2] - pad_top) / omega_scale_y
            + geometry["crop_top"]
        ) * ufo_scale_y
    return transformed, geometries


def intrinsics_metrics(predicted, target, image_size):
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    height, width = image_size
    focal_relative = np.concatenate([
        np.abs(predicted[:, 0, 0] / target[:, 0, 0] - 1),
        np.abs(predicted[:, 1, 1] / target[:, 1, 1] - 1),
    ])
    predicted_fov = np.stack([
        2 * np.arctan((width / 2) / predicted[:, 0, 0]),
        2 * np.arctan((height / 2) / predicted[:, 1, 1]),
    ], axis=1)
    target_fov = np.stack([
        2 * np.arctan((width / 2) / target[:, 0, 0]),
        2 * np.arctan((height / 2) / target[:, 1, 1]),
    ], axis=1)
    return {
        "focal_relative_error_percent_mean": float(focal_relative.mean() * 100),
        "fov_error_deg_mean": float(np.degrees(np.abs(predicted_fov - target_fov)).mean()),
        "principal_point_error_px_mean": float(np.linalg.norm(
            predicted[:, :2, 2] - target[:, :2, 2], axis=1
        ).mean()),
    }


def unique_preprocessing_geometries(geometries):
    unique = []
    for geometry in geometries:
        if geometry not in unique:
            unique.append(geometry)
    return unique


def umeyama_similarity(source, target):
    """Return s, R, t for target ~= s * R @ source + t."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"expected matching [N,3] points, got {source.shape}, {target.shape}")
    if len(source) < 3:
        raise ValueError("at least three camera centers are required for Sim(3) alignment")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        signs[-1] = -1
    rotation = u @ np.diag(signs) @ vt
    variance = np.sum(source_centered**2) / len(source)
    if variance < 1e-12:
        raise ValueError("degenerate camera centers cannot determine Sim(3)")
    scale = float(np.sum(singular_values * signs) / variance)
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def orientation_aware_similarity(source_c2w, target_c2w):
    """Fit Sim(3), using camera orientations to resolve degenerate trajectories."""
    source_c2w = np.asarray(source_c2w, dtype=np.float64)
    target_c2w = np.asarray(target_c2w, dtype=np.float64)
    if source_c2w.shape != target_c2w.shape or source_c2w.ndim != 3:
        raise ValueError(
            f"expected matching [N,4,4] poses, got {source_c2w.shape}, "
            f"{target_c2w.shape}"
        )
    if source_c2w.shape[1:] != (4, 4) or len(source_c2w) < 2:
        raise ValueError("at least two homogeneous camera poses are required")

    # Each target rotation should equal global_rotation @ source_rotation.
    cross_rotation = np.einsum(
        "nij,nkj->ik",
        target_c2w[:, :3, :3],
        source_c2w[:, :3, :3],
    )
    u, _, vt = np.linalg.svd(cross_rotation)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        signs[-1] = -1
    rotation = u @ np.diag(signs) @ vt

    source = source_c2w[:, :3, 3]
    target = target_c2w[:, :3, 3]
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    rotated_source = (rotation @ (source - source_mean).T).T
    target_centered = target - target_mean
    denominator = np.sum(rotated_source**2)
    if denominator < 1e-12:
        raise ValueError("degenerate camera centers cannot determine Sim(3) scale")
    scale = float(np.sum(rotated_source * target_centered) / denominator)
    if scale <= 0:
        raise ValueError(f"fitted Sim(3) scale must be positive, got {scale}")
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def apply_similarity(c2w, scale, rotation, translation):
    aligned = np.asarray(c2w, dtype=np.float64).copy()
    aligned[:, :3, :3] = np.einsum("ij,njk->nik", rotation, c2w[:, :3, :3])
    aligned[:, :3, 3] = (
        scale * np.einsum("ij,nj->ni", rotation, c2w[:, :3, 3]) + translation
    )
    return aligned


def rotation_degrees(left, right):
    relative = np.einsum("nij,nkj->nik", left, right)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cosine))


def pose_metrics(predicted, target, frame_ids, camera_ids):
    rotation_error = rotation_degrees(predicted[:, :3, :3], target[:, :3, :3])
    translation_error = np.linalg.norm(predicted[:, :3, 3] - target[:, :3, 3], axis=1)
    rpe_rotation = []
    direction_error = []
    rpe_translation = []
    for camera_id in sorted(set(camera_ids.tolist())):
        indices = np.flatnonzero(camera_ids == camera_id)
        indices = indices[np.argsort(frame_ids[indices])]
        for first, second in zip(indices[:-1], indices[1:]):
            pred_relative = np.linalg.inv(predicted[first]) @ predicted[second]
            gt_relative = np.linalg.inv(target[first]) @ target[second]
            rpe_rotation.append(
                rotation_degrees(
                    pred_relative[None, :3, :3], gt_relative[None, :3, :3]
                )[0]
            )
            pred_delta = predicted[second, :3, 3] - predicted[first, :3, 3]
            gt_delta = target[second, :3, 3] - target[first, :3, 3]
            denominator = np.linalg.norm(pred_delta) * np.linalg.norm(gt_delta)
            if denominator > 1e-12:
                cosine = np.clip(pred_delta @ gt_delta / denominator, -1, 1)
                direction_error.append(np.degrees(np.arccos(cosine)))
            rpe_translation.append(np.linalg.norm(pred_relative[:3, 3] - gt_relative[:3, 3]))
    return {
        "rotation_error_deg_mean": float(np.mean(rotation_error)),
        "rotation_error_deg_median": float(np.median(rotation_error)),
        "translation_error_m_mean": float(np.mean(translation_error)),
        "sim3_ate_rmse_m": float(np.sqrt(np.mean(translation_error**2))),
        "rpe_rotation_deg_mean": float(np.mean(rpe_rotation)),
        "rpe_translation_m_mean": float(np.mean(rpe_translation)),
        "translation_direction_error_deg_mean": float(np.mean(direction_error)),
    }


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    entries = [
        item for item in manifest["images"]
        if args.scope == "all" or item["role"] == "context"
    ]
    image_paths = [item["path"] for item in entries]
    missing = [path for path in image_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"manifest contains missing images: {missing[:3]}")

    images = load_and_preprocess_images(
        image_paths, image_resolution=args.image_resolution
    ).cuda()
    model = VGGTOmega().eval()
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.cuda()
    del state_dict

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictions = model(images)
        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:]
        )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started

    raw_w2c = as_homogeneous(extrinsics[0].detach().float().cpu().numpy())
    raw_c2w = np.linalg.inv(raw_w2c)
    gt_c2w = np.asarray([item["gt_c2w_opencv"] for item in entries], dtype=np.float64)
    gt_native = np.asarray([item["gt_camera_to_world"] for item in entries], dtype=np.float64)
    scale, rotation, translation = orientation_aware_similarity(raw_c2w, gt_c2w)
    aligned_c2w = apply_similarity(raw_c2w, scale, rotation, translation)
    aligned_w2c = np.linalg.inv(aligned_c2w)
    opencv_to_dataset = np.asarray(manifest["opencv_to_dataset"], dtype=np.float64)
    aligned_native = aligned_c2w @ np.linalg.inv(opencv_to_dataset)
    predicted_intrinsics = intrinsics[0].detach().float().cpu().numpy().astype(np.float32)
    predicted_intrinsics_ufo, preprocessing_geometry = transform_intrinsics_to_ufo(
        predicted_intrinsics,
        image_paths,
        manifest["ufo_image_size"],
        image_resolution=args.image_resolution,
    )
    gt_intrinsics_ufo = np.asarray(
        [item["gt_intrinsics_ufo"] for item in entries], dtype=np.float64
    )
    frame_ids = np.asarray([item["frame_id"] for item in entries], dtype=np.int32)
    camera_ids = np.asarray([item["camera_id"] for item in entries])
    roles = np.asarray([item["role"] for item in entries])

    scene_output = args.output_dir / manifest["scene_name"]
    scene_output.mkdir(parents=True, exist_ok=True)
    output_path = scene_output / "omega_pose_override.npz"
    np.savez_compressed(
        output_path,
        scene_name=np.asarray(manifest["scene_name"]),
        scope=np.asarray(args.scope),
        frame_ids=frame_ids,
        camera_ids=camera_ids,
        roles=roles,
        omega_w2c_raw=raw_w2c.astype(np.float32),
        omega_c2w_raw=raw_c2w.astype(np.float32),
        omega_w2c_aligned=aligned_w2c,
        omega_c2w_aligned=aligned_c2w,
        omega_camera_to_world_aligned=aligned_native,
        gt_c2w=gt_c2w,
        gt_camera_to_world=gt_native,
        predicted_intrinsics=predicted_intrinsics,
        predicted_intrinsics_ufo=predicted_intrinsics_ufo.astype(np.float32),
        gt_intrinsics_ufo=gt_intrinsics_ufo,
        sim3_scale=np.asarray(scale),
        sim3_rotation=rotation,
        sim3_translation=translation,
    )
    metrics = {
        "scene_name": manifest["scene_name"],
        "scope": args.scope,
        "num_images": len(entries),
        "num_context_images": sum(item["role"] == "context" for item in entries),
        "num_target_images": sum(item["role"] == "target" for item in entries),
        "sim3": {
            "method": "orientation_aware",
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
        "pose": pose_metrics(aligned_c2w, gt_c2w, frame_ids, camera_ids),
        "intrinsics": intrinsics_metrics(
            predicted_intrinsics_ufo, gt_intrinsics_ufo, manifest["ufo_image_size"]
        ),
        "omega_image_size": list(predictions["images"].shape[-2:]),
        "ufo_image_size": manifest["ufo_image_size"],
        "preprocessing_geometries": unique_preprocessing_geometries(
            preprocessing_geometry
        ),
        "inference_seconds": inference_seconds,
        "peak_gpu_memory_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "output": str(output_path.resolve()),
    }
    (scene_output / "pose_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
