# Pose-free camera preprocessing: runtime snapshot

This is a source snapshot of the installed UFO/Waymo preprocessing pipeline,
not yet a standalone pip package or a new camera estimation model. Captured
working files include uncommitted changes. See runtime_inventory.json for
SHA-256 hashes, dependency versions, source roots, revisions and missing files.

## Current pipeline

RGB manifest -> VGGT-Omega pose/K/depth/confidence -> MoGe-2 + GCA scale
-> metric window cameras -> overlap SE(3) alignment -> full-scene camera NPZ.

The four-GPU scheduler shares a scene queue. Each GPU processes one independent
scene at a time. Models are reused over a window group; this is not DDP training
or a 179-window neural-network forward batch. The installed backend remains
the authority for inference numerics and cleanup.

## What was copied

- h200_adapter/: scheduling, accelerated stages, configuration and UFO reader.
- source_ufo/: RGB manifest and reference camera/scale/alignment exporters.
- omega_adapter/: the existing intrinsic-coordinate conversion helper only.
- dependency_notices/: local model-repository notices; model implementations
  and weights are external dependencies.
- camera_npz_schema.json: key names, shapes/dtypes, and selected convention
  labels from one validated final camera file, when available; no camera arrays.
- runtime_status.json: a time-stamped inspection, not a live dashboard.

Datasets, RGB images, SAM tracks, weights, camera NPZ contents, raw depth arrays,
virtual environments, credentials and original .git directories are not copied.
Only explicitly listed source files are read; this is not a recursive repo copy.
Inspect hardcoded server paths before making this snapshot public. No remote
repository is created and nothing is pushed by the collection tool.

## Current inputs

- UFO-format Waymo RGB images and a scene annotation list.
- Scene JSON metadata: dataset, scene_id, scene_name, fps, num_timesteps,
  relative_image_path indexed by camera ID and frame.
- Camera selection and target image size from config. The current protocol
  uses cameras 1/0/2 and target K at height 160, width 240.
- Omega and MoGe-2 source checkouts, local checkpoints and an installed GPU
  Python environment. No GT camera calibration, poses, depth or LiDAR are
  inputs to the RGB-only camera prediction chain; SAM is used later by UFO.

The legacy annotation JSON may contain GT fields; the RGB-only manifest exporter
selects metadata and RGB paths. It does not use those GT fields for prediction.
The current camera protocol is all_rgb: both context and target RGB enter the
offline camera estimator. It must not be described as context-only or online.

## Current outputs

cache/global_aligned/<scene_name>/omega_pose_override.npz contains frame_ids,
camera_ids, omega_c2w_global_metric (OpenCV camera basis),
omega_camera_to_world_global_metric (dataset camera basis),
predicted_intrinsics_ufo and convention labels. Resolve records by the pair
(frame_id, camera_id), not by assumed input ordering. Each scene has its own
reference frame; global_metric does not mean a GPS/world-map coordinate frame.
Metric scale is estimated by MoGe-2/GCA, not measured or guaranteed exact.

The intrinsics are in pixels for the target image size recorded in the cache
contract. Final NPZs do not contain RGB, masks, depth maps, a mesh or Gaussians.
The file ../.posefree_contract.json records the image size and camera protocol.

Intermediate raw and scale files are removed after successful metric export;
window poses and manifests after successful alignment. Interrupted work may
leave residual files. Final scene cameras and logs remain available.

## Proposed reusable plugin boundary (not implemented by this snapshot)

1. Separate a generic RGB manifest API from Waymo/UFO dataset adapters.
2. Keep Omega, MoGe and scale/alignment backends external and versioned.
3. Make GPU count, reference camera, window length/stride, image geometry,
   output path and cleanup policy configurable.
4. Expose prepare, status, validate and export commands plus a Python API.
5. Export an OpenCV camera-to-world array and per-image intrinsics with explicit
   coordinate conventions, image transforms, frame/camera keys and provenance.
6. Keep a UFO output adapter for the existing omega_pose_override.npz schema.
7. Publish CPU schema tests and GPU equivalence/throughput results separately.
8. Choose the wrapper's license deliberately and retain upstream notices.
   Do not relicense or bundle third-party model weights as wrapper-owned code.

Suggested public description: "Offline RGB-only camera preprocessing with
VGGT-Omega, MoGe-2/GCA scale estimation and overlap alignment, with a UFO/Waymo
adapter." Do not present the dependencies as newly proposed camera models.

Before release, compare this snapshot with the source used for the completed
798-scene run. Audit missing imported modules and replace user-specific paths
with configuration. The manifest API and general data adapters are still work
to implement; this snapshot itself is not a drop-in universal reconstruction
plugin.
