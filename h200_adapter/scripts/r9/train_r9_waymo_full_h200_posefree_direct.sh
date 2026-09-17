#!/usr/bin/env bash
set -euo pipefail

# Independent H200 pose-free run. This never uses the GT-UFO run directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

PYTHON="${UFO_PYTHON_BIN:-/root/miniconda3/envs/dggt_data/bin/python}"
TORCHRUN="${UFO_TORCHRUN_BIN:-/root/miniconda3/envs/dggt_data/bin/torchrun}"
DATA_ROOT="${UFO_DATA_ROOT:-${ROOT}/data/UFO_paper}"
POSE_ROOT="${UFO_POSEFREE_CAMERA_ROOT:-${ROOT}/outputs/posefree_camera_full/global_aligned}"
RUN_DIR="${ROOT}/outputs/posefree_100k/r9_waymo_full_posefree_100k"

export PYTHONUNBUFFERED=1

if [[ ! -d "${POSE_ROOT}" ]]; then
  echo "ERROR: full-scene pose-free camera cache directory does not exist: ${POSE_ROOT}" >&2
  echo "It must contain <scene_name>/omega_pose_override.npz for all 798 Waymo train scenes." >&2
  exit 2
fi

"${PYTHON}" tools/r9/check_posefree_full_camera_cache.py \
  --data-root "${DATA_ROOT}" \
  --annotation-file "${DATA_ROOT}/scene_list/waymo_train.txt" \
  --cache-root "${POSE_ROOT}"

# Keep the H200 compatibility and SAM completion checks used by the GT run.
bash tools/r9/assert_r9_h200_batch_compat.sh

"${PYTHON}" - "${DATA_ROOT}" <<'PY'
import json
import sys
from pathlib import Path
root = Path.cwd()
data_root = Path(sys.argv[1])
sam_root = root / "data/r9_sam_tracks"
lines = [x.strip() for x in (data_root / "scene_list/waymo_train.txt").read_text().splitlines() if x.strip()]
assert len(lines) == 798, f"Expected 798 scenes, got {len(lines)}"
missing = []
count = 0
for item in lines:
    p = Path(item)
    if not p.is_absolute(): p = data_root / p
    scene = json.loads(p.read_text())["scene_name"]
    for camera in ("1", "0", "2"):
        marker = sam_root / scene / camera / ".r9_sam2_done.json"
        if not marker.is_file(): missing.append(str(marker)); continue
        if int(json.loads(marker.read_text()).get("mask_count", -1)) <= 0:
            missing.append(str(marker) + " [invalid mask_count]"); continue
        count += 1
print(f"SAM_DONE_MARKERS={count}/2394", flush=True)
if missing:
    print("Missing/invalid examples:", *missing[:20], sep="\n", flush=True)
    raise SystemExit(2)
PY

"${PYTHON}" tools/r9/check_full_start_mode.py --run-dir "${RUN_DIR}"

echo "Starting independent 8xH200 pose-free training"
echo "run_dir=${RUN_DIR}"
echo "batch_per_gpu=4 global_batch=64 iterations=100000"

source "${ROOT}/scripts/h200/env_h200_offline.sh"

exec "${TORCHRUN}" \
  --standalone \
  --nproc_per_node=8 \
  main.py \
  --config configs/h200/r9_waymo_full_posefree_100k.json \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --ddp_accumulation_no_sync \
  --auto_resume \
  --num_iterations 100000 \
  --project posefree_100k \
  --exp_name r9_waymo_full_posefree_100k \
  --output_dir outputs \
  --pose_override_dir "${POSE_ROOT}" \
  --intrinsics_override_dir "${POSE_ROOT}" \
  --log_every_n_iters 10 \
  --skip_initial_validation \
  --skip_final_evaluation
