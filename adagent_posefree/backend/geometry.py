"""Original RGB geometry helpers; no GT loader or training imports."""
import numpy as np

TOP_N = 3
CONF_PERCENTILE = 0.5
OMEGA_RESOLUTION = 512
PATCH_SIZE = 16

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
    import torch
    import torch.nn.functional as F
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

def transform_intrinsics_to_ufo(
    intrinsics, image_paths, ufo_image_size, image_resolution=512, patch_size=16
):
    """Map K from Omega's crop/resize/pad tensor into UFO image pixels."""
    from vggt_omega.utils.load_fn import _balanced_target_shape
    from PIL import Image
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
