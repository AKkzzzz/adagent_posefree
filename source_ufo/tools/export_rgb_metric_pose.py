#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import numpy as np


p = argparse.ArgumentParser()
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--omega-npz", type=Path, required=True)
p.add_argument("--scale-json", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()

manifest = json.loads(args.manifest.read_text())
scale_report = json.loads(args.scale_json.read_text())

scale = float(scale_report["global_scale"])
if not np.isfinite(scale) or scale <= 0:
    raise ValueError(f"invalid scale: {scale}")

with np.load(args.omega_npz, allow_pickle=False) as x:
    frame_ids = x["frame_ids"].astype(np.int32)
    camera_ids = x["camera_ids"].astype(str)
    roles = x["roles"].astype(str)
    raw_c2w = x["omega_c2w_raw"].astype(np.float64)
    raw_w2c = x["omega_w2c_raw"].astype(np.float64)
    K = x["predicted_intrinsics_ufo"].astype(np.float64)

scene_name = manifest["scene_name"]

# 与原 Omega exporter 完全相同的 gauge：
# 第一个 timestamp 的 front camera 作为 identity。
first_frame = int(frame_ids.min())
idx = np.flatnonzero(
    (frame_ids == first_frame) & (camera_ids == "0")
)

if len(idx) != 1:
    raise ValueError(
        f"expected exactly one front camera at frame {first_frame}"
    )

world_to_local = np.linalg.inv(raw_c2w[int(idx[0])])

local_c2w = np.einsum(
    "ij,njk->nik",
    world_to_local,
    raw_c2w
)

# 唯一新操作：
# 用 RGB correspondence + MoGe 得到的 metric scale
local_c2w[:, :3, 3] *= scale

local_w2c = np.linalg.inv(local_c2w)

opencv_to_dataset = np.asarray(
    manifest["opencv_to_dataset"],
    dtype=np.float64
)

# PoseOverrideStore 在 rig_local_metric 下读取这个字段
local_native = (
    local_c2w @ np.linalg.inv(opencv_to_dataset)
)

identity_error = float(
    np.abs(local_c2w[int(idx[0])] - np.eye(4)).max()
)

args.output.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    args.output,
    scene_name=np.asarray(scene_name),
    scope=np.asarray("all"),

    coordinate_frame=np.asarray("rig_local_metric"),
    metric_scale_source=np.asarray(
        "moge2_rgb_correspondence_global_scale"
    ),
    world_gauge=np.asarray(
        "first_timestamp_front_camera"
    ),

    frame_ids=frame_ids,
    camera_ids=camera_ids,
    roles=roles,

    omega_w2c_raw=raw_w2c.astype(np.float32),
    omega_c2w_raw=raw_c2w.astype(np.float32),

    omega_w2c_rig_local=local_w2c.astype(np.float32),
    omega_c2w_rig_local=local_c2w.astype(np.float32),

    omega_camera_to_world_rig_local=
        local_native.astype(np.float32),

    predicted_intrinsics_ufo=K.astype(np.float32),

    rgb_metric_scale=np.asarray(scale),
    gauge_frame_id=np.asarray(first_frame),
    gauge_camera_id=np.asarray("0"),
)

print("========================================")
print("RGB-only metric pose exported")
print("scene          :", scene_name)
print("metric scale   :", scale)
print("scale source   : MoGe-2 + RGB correspondence")
print("GT geometry    : NOT USED")
print("front0 error   :", identity_error)
print("output         :", args.output)
print("========================================")
