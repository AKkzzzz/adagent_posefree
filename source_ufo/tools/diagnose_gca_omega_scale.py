#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


TOP_N = 3
CONF_PERCENTILE = 0.5
OMEGA_RESOLUTION = 512
PATCH_SIZE = 16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--omega-repo", type=Path, required=True)
    p.add_argument("--omega-checkpoint", type=Path, required=True)
    p.add_argument("--moge-repo", type=Path, required=True)
    p.add_argument("--moge-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def omega_crop(arr):
    h, w = arr.shape[-2:]
    aspect = h / float(w)

    if aspect < 0.5:
        crop_w = min(w, max(1, int(round(h / 0.5))))
        left = max((w - crop_w) // 2, 0)
        return arr[..., :, left:left + crop_w]

    if aspect > 2.0:
        crop_h = min(h, max(1, int(round(w * 2.0))))
        top = max((h - crop_h) // 2, 0)
        return arr[..., top:top + crop_h, :]

    return arr


def omega_balanced_shape(h, w):
    aspect = h / float(w)
    token_number = (OMEGA_RESOLUTION // PATCH_SIZE) ** 2

    w_patches_float = np.sqrt(token_number / aspect)
    h_patches_float = token_number / w_patches_float

    wp = max(1, int(np.round(w_patches_float)))
    hp = max(1, int(np.round(h_patches_float)))

    return hp * PATCH_SIZE, wp * PATCH_SIZE


def transform_to_omega(arr, out_hw, is_mask=False):
    x = torch.as_tensor(arr).float()
    x = omega_crop(x)

    h, w = x.shape[-2:]
    th, tw = omega_balanced_shape(h, w)

    x = x[None, None]

    x = F.interpolate(
        x,
        size=(th, tw),
        mode="nearest" if is_mask else "bilinear",
        align_corners=None if is_mask else False,
    )

    out_h, out_w = out_hw

    pad_h = out_h - th
    pad_w = out_w - tw

    if pad_h < 0 or pad_w < 0:
        raise RuntimeError(
            f"target {th}x{tw} larger than Omega output {out_h}x{out_w}"
        )

    pt = pad_h // 2
    pb = pad_h - pt
    pl = pad_w // 2
    pr = pad_w - pl

    if pad_h or pad_w:
        x = F.pad(x, (pl, pr, pt, pb), value=0)

    x = x[0, 0]

    if is_mask:
        return x > 0.5

    return x


def main():
    args = parse_args()

    manifest = json.loads(args.manifest.read_text())
    entries = manifest["images"]
    paths = [e["path"] for e in entries]

    device = "cuda"

    # ==========================================================
    # VGGT-Omega reconstruction
    # ==========================================================
    sys.path.insert(0, str(args.omega_repo))

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    print(f"[Omega] running {len(paths)} images", flush=True)

    images = load_and_preprocess_images(
        paths,
        image_resolution=OMEGA_RESOLUTION,
    ).to(device)

    omega = VGGTOmega().eval().to(device)

    state = torch.load(
        args.omega_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    omega.load_state_dict(state)
    del state

    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16
    ):
        pred = omega(images)

    depth = pred["depth"][0].float().cpu()
    conf = pred["depth_conf"][0].float().cpu()

    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]

    del omega, pred, images
    torch.cuda.empty_cache()

    if depth.ndim != 3:
        raise RuntimeError(f"unexpected Omega depth shape {depth.shape}")

    N, OH, OW = depth.shape

    if N != len(entries):
        raise RuntimeError(
            f"Omega images={N}, manifest images={len(entries)}"
        )

    # ==========================================================
    # GCA Step 1:
    # rank reconstruction images by confidence score
    # ==========================================================
    scores = []

    for i in range(N):
        c = conf[i]

        threshold = torch.quantile(
            c.float(),
            CONF_PERCENTILE,
        )

        mask = c > threshold
        score = c[mask].sum().item()

        scores.append((score, i))

    scores.sort(reverse=True)

    selected = [
        idx for _, idx in scores[:min(TOP_N, N)]
    ]

    print("[GCA] selected Top-3:", selected, flush=True)

    # ==========================================================
    # MoGe-2
    # ==========================================================
    sys.path.insert(0, str(args.moge_repo))
    from moge.model.v2 import MoGeModel

    moge = MoGeModel.from_pretrained(
        args.moge_model
    ).eval().to(device)

    all_ratios = []
    rows = []

    # ==========================================================
    # Exact GCA rule:
    # scale = median(metric_depth / relative_depth)
    # ==========================================================
    for idx in selected:
        e = entries[idx]

        rgb_np = np.asarray(
            Image.open(e["path"]).convert("RGB")
        ).copy()

        rgb = (
            torch.from_numpy(rgb_np)
            .float()
            .permute(2, 0, 1)
            .to(device)
            / 255.0
        )

        with torch.inference_mode():
            out = moge.infer(
                rgb,
                resolution_level=9,
                use_fp16=True,
                apply_mask=False,
            )

        metric_depth = out["depth"].float().cpu()
        metric_mask = out["mask"].cpu().bool()

        metric_depth = transform_to_omega(
            metric_depth,
            (OH, OW),
            is_mask=False,
        )

        metric_mask = transform_to_omega(
            metric_mask,
            (OH, OW),
            is_mask=True,
        )

        relative_depth = depth[idx]
        relative_conf = conf[idx]

        threshold = torch.quantile(
            relative_conf.float(),
            CONF_PERCENTILE,
        )

        relative_mask = relative_conf > threshold

        valid = (
            relative_mask
            & metric_mask
            & torch.isfinite(metric_depth)
            & torch.isfinite(relative_depth)
            & (relative_depth > 1e-4)
            & (metric_depth > 1e-4)
        )

        n = int(valid.sum())

        if n < 100:
            print(
                f"[skip] idx={idx}, valid={n}",
                flush=True,
            )
            continue

        ratios = (
            metric_depth[valid]
            / relative_depth[valid]
        ).float()

        image_scale = torch.median(ratios).item()

        row = {
            "index": idx,
            "frame": int(e["frame_id"]),
            "camera": str(e["camera_id"]),
            "confidence_score": float(scores[
                [x[1] for x in scores].index(idx)
            ][0]),
            "valid_pixels": n,
            "median_scale": image_scale,
        }

        rows.append(row)
        all_ratios.append(ratios)

        print(
            f"idx={idx:02d} "
            f"frame={int(e['frame_id']):03d} "
            f"cam={e['camera_id']} | "
            f"valid={n:6d} | "
            f"scale={image_scale:.6f}",
            flush=True,
        )

    if not all_ratios:
        raise RuntimeError("GCA scale estimation failed")

    final_ratios = torch.cat(all_ratios)

    # Exact official GCA estimator
    scale = torch.median(final_ratios).item()

    logs = torch.log(final_ratios)
    med_log = torch.median(logs)
    log_mad = torch.median(
        torch.abs(logs - med_log)
    ).item()

    report = {
        "method": "gca_metric_scale_adapted_to_vggt_omega",
        "source_method": "GCA estimate_scale",
        "top_n": TOP_N,
        "relative_conf_percentile": CONF_PERCENTILE,
        "global_scale": float(scale),
        "global_log_mad": float(log_mad),
        "selected": rows,
        "num_ratio_pixels": int(final_ratios.numel()),
        "uses_gt_geometry": False,
        "uses_gt_pose": False,
        "uses_gt_intrinsics": False,
        "uses_gt_depth": False,
        "uses_camera_to_ego": False,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print()
    print("================ GCA-OMEGA ================")
    print("scale       :", scale)
    print("log MAD     :", log_mad)
    print("ratio pixels:", final_ratios.numel())
    print("selected    :", rows)
    print("GT geometry : NOT USED")
    print("===========================================")


if __name__ == "__main__":
    main()
