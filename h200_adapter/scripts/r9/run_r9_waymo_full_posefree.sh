#!/usr/bin/env bash
set -euo pipefail

# One repository entry point with two physical stages:
#   prepare : run from the 4090-visible environment and write the shared cache
#   train   : run from the H200 container and consume only that shared cache
# `all` is useful only when both environments are visible to one process.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:-train}"
DATA_ROOT="${UFO_DATA_ROOT:-${ROOT}/data/UFO_paper}"
CACHE_ROOT="${UFO_POSEFREE_CACHE_ROOT:-${ROOT}/outputs/posefree_camera_full}"

case "${STAGE}" in
  prepare)
    SOURCE_ROOT="${UFO_POSEFREE_SOURCE_ROOT:?Set UFO_POSEFREE_SOURCE_ROOT to the 4090-visible pose-free root (containing pose-freeufo, vggt-omega and moge).}"
    SOURCE_UFO_ROOT="${UFO_POSEFREE_UFO_ROOT:-${SOURCE_ROOT}/pose-freeufo}"
    PREP_PYTHON="${UFO_POSEFREE_PREP_PYTHON:-${SOURCE_ROOT}/.venv-moge2/bin/python}"
    OMEGA_REPO="${UFO_POSEFREE_OMEGA_REPO:-${SOURCE_ROOT}/vggt-omega}"
    OMEGA_CHECKPOINT="${UFO_POSEFREE_OMEGA_CHECKPOINT:-${OMEGA_REPO}/checkpoints/vggt_omega_1b_512.pt}"
    MOGE_REPO="${UFO_POSEFREE_MOGE_REPO:-${SOURCE_ROOT}/moge}"
    MOGE_MODEL="${UFO_POSEFREE_MOGE_MODEL:-${SOURCE_ROOT}/checkpoints/moge-2-vitl/model.pt}"
    test -x "${PREP_PYTHON}" || { echo "Missing 4090 preparation Python: ${PREP_PYTHON}" >&2; exit 2; }
    exec "${PREP_PYTHON}" "${ROOT}/tools/r9/prepare_posefree_waymo_full.py" \
      --source-ufo-root "${SOURCE_UFO_ROOT}" \
      --config "${ROOT}/configs/h200/r9_waymo_full_posefree_100k.json" \
      --data-root "${DATA_ROOT}" \
      --annotation-file "${DATA_ROOT}/scene_list/waymo_train.txt" \
      --cache-root "${CACHE_ROOT}" \
      --python "${PREP_PYTHON}" \
      --omega-repo "${OMEGA_REPO}" \
      --omega-checkpoint "${OMEGA_CHECKPOINT}" \
      --moge-repo "${MOGE_REPO}" \
      --moge-model "${MOGE_MODEL}" \
      --scene-first "${UFO_POSEFREE_SCENE_FIRST:-0}" \
      --scene-last "${UFO_POSEFREE_SCENE_LAST:-797}" \
      --window-batch-size "${UFO_POSEFREE_WINDOW_BATCH_SIZE:-179}" \
      ${UFO_POSEFREE_FORCE:+--force}
    ;;
  train)
    export UFO_POSEFREE_CAMERA_ROOT="${UFO_POSEFREE_CAMERA_ROOT:-${CACHE_ROOT}/global_aligned}"
    exec "${ROOT}/scripts/r9/train_r9_waymo_full_h200_posefree_direct.sh"
    ;;
  all)
    "${BASH_SOURCE[0]}" prepare
    "${BASH_SOURCE[0]}" train
    ;;
  *)
    echo "Usage: $0 {prepare|train|all}" >&2
    exit 2
    ;;
esac
