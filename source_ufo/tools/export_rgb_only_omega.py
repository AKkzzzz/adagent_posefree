#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def homo(extri):
    n = len(extri)
    out = np.tile(np.eye(4), (n,1,1))
    out[:, :3, :4] = extri
    return out


p = argparse.ArgumentParser()
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--omega-repo", type=Path, required=True)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()

manifest = json.load(open(args.manifest))
entries = manifest["images"]
paths = [e["path"] for e in entries]

sys.path.insert(0, str(args.omega_repo))
sys.path.insert(0, str(args.omega_repo / "tools"))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from export_ufo_pose_override import transform_intrinsics_to_ufo

device = "cuda"

images = load_and_preprocess_images(
    paths,
    image_resolution=512
).to(device)

model = VGGTOmega().eval().to(device)
state = torch.load(
    args.checkpoint,
    map_location="cpu",
    weights_only=True
)
model.load_state_dict(state)

with torch.inference_mode(), torch.autocast(
    "cuda", dtype=torch.bfloat16
):
    pred = model(images)
    extri, intri = encoding_to_camera(
        pred["pose_enc"],
        pred["images"].shape[-2:]
    )

    omega_depth = pred["depth"][0].float().cpu().numpy()
    omega_depth_conf = pred["depth_conf"][0].float().cpu().numpy()

w2c = homo(extri[0].float().cpu().numpy())
c2w = np.linalg.inv(w2c)

K = intri[0].float().cpu().numpy()

K_ufo, _ = transform_intrinsics_to_ufo(
    K,
    paths,
    manifest["ufo_image_size"],
    image_resolution=512,
)

frame_ids = np.asarray(
    [e["frame_id"] for e in entries],
    dtype=np.int32
)
camera_ids = np.asarray(
    [e["camera_id"] for e in entries]
)
roles = np.asarray(
    [e["role"] for e in entries]
)

args.output.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    args.output,

    scene_name=np.asarray(manifest["scene_name"]),
    scope=np.asarray("all"),

    frame_ids=frame_ids,
    camera_ids=camera_ids,
    roles=roles,

    omega_w2c_raw=w2c.astype(np.float32),
    omega_c2w_raw=c2w.astype(np.float32),

    predicted_intrinsics_ufo=K_ufo.astype(np.float32),

    # Cached for exact GCA scale estimation without a second Omega pass.
    omega_depth_raw=omega_depth.astype(np.float32),
    omega_depth_conf_raw=omega_depth_conf.astype(np.float32),
)

print("RGB-only Omega exported:", args.output)
print("images:", len(entries))
print("GT geometry: NOT USED")
