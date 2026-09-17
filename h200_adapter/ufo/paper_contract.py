"""Paper-explicit UFO contracts shared by data, recurrence, and decoding.

This module intentionally excludes details the paper does not define, including
attention masks, flow prediction, MLP hidden widths, depth normalization, and
the timing of training supervision.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Preprocessed Waymo camera IDs: FRONT_LEFT, FRONT, FRONT_RIGHT.
WAYMO_CAMERAS = ("1", "0", "2")
WAYMO_IMAGE_SIZE = (160, 240)
CONTEXT_STRIDE = 5
VISIBLE_TOKEN_BUDGET = 3600
SCENE_FEATURE_DIM = 768
GAUSSIANS_PER_TOKEN = 64
MAX_TRACKED_BOXES = 32


@dataclass(frozen=True)
class FrameProtocol:
    context: tuple[int, ...]
    supervision: tuple[int, ...]


def split_context_supervision(start: int, end: int, stride: int = CONTEXT_STRIDE) -> FrameProtocol:
    """Split a half-open frame range according to the paper's Waymo protocol."""
    if not 0 <= start < end:
        raise ValueError(f"expected 0 <= start < end, got {start=}, {end=}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    context = tuple(range(start, end, stride))
    context_set = set(context)
    supervision = tuple(i for i in range(start, end) if i not in context_set)
    return FrameProtocol(context=context, supervision=supervision)


def relative_se3(global_c2w: torch.Tensor, local_c2w: torch.Tensor) -> torch.Tensor:
    """Return the transform mapping current-local coordinates into scene-world coordinates."""
    if global_c2w.shape[-2:] != (4, 4) or local_c2w.shape[-2:] != (4, 4):
        raise ValueError("camera transforms must end in [4, 4]")
    return global_c2w @ torch.linalg.inv(local_c2w)


def transform_points(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply an SE(3) transform to row-vector points."""
    return (
        torch.matmul(transform[..., :3, :3], points.unsqueeze(-1)).squeeze(-1)
        + transform[..., :3, 3]
    )


def transform_directions(transform: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Apply only the rotational component of an SE(3) transform."""
    return torch.matmul(
        transform[..., :3, :3], directions.unsqueeze(-1)
    ).squeeze(-1)


def expand_token_assignments(token_weights: torch.Tensor, gaussians_per_token: int) -> torch.Tensor:
    """Make every Gaussian decoded by one scene token share its object assignment."""
    if token_weights.ndim != 3:
        raise ValueError("token_weights must be [B, N_token, N_object]")
    if gaussians_per_token <= 0:
        raise ValueError("gaussians_per_token must be positive")
    return token_weights.repeat_interleave(gaussians_per_token, dim=1)


def split_aux_tokens(
    aux: torch.Tensor, num_cameras: int, num_motion_tokens: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the recurrent sky, per-camera affine, and optional legacy motion tokens."""
    required = 1 + num_cameras + num_motion_tokens
    if aux.ndim != 3 or aux.shape[1] < required:
        raise ValueError(f"expected [B, >= {required}, D] auxiliary tokens")
    sky = aux[:, :1]
    affine = aux[:, 1:1 + num_cameras]
    motion = aux[:, 1 + num_cameras:required]
    return sky, affine, motion


def assert_paper_training_ready(args) -> None:
    """Prevent formal paper training while required undisclosed contracts are unresolved."""
    if not getattr(args, "paper_frame_protocol", False):
        return
    blockers = []
    if getattr(args, "paper_supervision_mode", "unknown") == "unknown":
        blockers.append("per-chunk versus final-scene supervision")
    if getattr(args, "enable_flow_reg_loss", False) and not getattr(
        args, "paper_forward_flow_impl", False
    ):
        blockers.append("forward-flow head")
    if blockers:
        raise RuntimeError(
            "Paper-contract training is blocked by unresolved semantics: "
            + ", ".join(blockers)
        )
