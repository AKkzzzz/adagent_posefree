#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${UFO_POSEFREE_SOURCE_ROOT:-/inspire/hdd3/project/intelligent-driving-agent/public/workspace/yx/ufoposefree}"
PREP_PYTHON="${UFO_POSEFREE_PREP_PYTHON:-${SOURCE}/.venv-moge2/bin/python}"
DATA="${UFO_DATA_ROOT:-${ROOT}/data/UFO_paper}"
CACHE="${UFO_POSEFREE_CACHE_ROOT:-/inspire/hdd/global_user/guoluosong-253108120129/yx-ufo/outputs/r9ufo/posefree_camera_full}"
test -x "${PREP_PYTHON}" || { echo "Missing Python: ${PREP_PYTHON}" >&2; exit 2; }

exec "${PREP_PYTHON}" -u "${ROOT}/tools/r9/prepare_posefree_multigpu.py" \
    --repo "${ROOT}" --source-root "${SOURCE}" \
    --data-root "${DATA}" --cache-root "${CACHE}" \
    --expected-gpus 4 \
    --gpus "${UFO_POSEFREE_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}" \
    --scene-first "${UFO_POSEFREE_SCENE_FIRST:-0}" \
    --scene-last "${UFO_POSEFREE_SCENE_LAST:-797}" \
    --window-batch-size "${UFO_POSEFREE_WINDOW_BATCH_SIZE:-179}" \
    --cpu-threads "${UFO_POSEFREE_CPU_THREADS:-4}" \
    --min-free-gib "${UFO_POSEFREE_MIN_FREE_GIB:-150}" "$@"
