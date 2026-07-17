"""
Shared liveness (passive anti-spoofing) preprocessing utilities.

Model: MiniFASNet from Minivision's Silent-Face-Anti-Spoofing (Apache-2.0).

Input convention (MUST match training-time transform, otherwise scores drift):

The reference is Minivision's official ``predict`` path
(``Silent-Face-Anti-Spoofing/src/anti_spoof_predict.py`` +
``src/utility.py::CropImage`` + ``src/data_io/transform.py``):

1. Crop — the raw *detection bbox* (x, y, w, h) is expanded around its own
   center by ``scale`` (2.7 for the 2.7_80x80 MiniFASNet variant): the crop
   width/height become ``w*scale`` / ``h*scale``. ``scale`` itself is first
   clamped so the expanded box never exceeds the source image
   (``scale = min((src_h-1)/h, (src_w-1)/w, scale)``). If the expanded box
   crosses an image border it is *shifted* back inside (not truncated), then
   clamped to ``[0, src-1]``. This is a faithful port of
   ``CropImage._get_new_box``.
2. Resize the crop to 80x80 (``cv2.resize``, default bilinear).
3. NO mean/variance normalization, NO /255 scaling: Minivision's transform
   pipeline for prediction is only their *custom* ``ToTensor``
   (``src/data_io/functional.py::to_tensor``), which converts HWC uint8 to a
   CHW *float* tensor while keeping the 0-255 value range — unlike
   torchvision's ToTensor it does NOT divide by 255, and there is no
   ``Normalize`` step at predict time.
4. Channel order stays **BGR** — images are read with ``cv2.imread`` and the
   official pipeline never converts to RGB.

So the canonical model input is: 80x80, BGR, float 0-255 (we hand backends a
uint8 BGR HWC crop; each backend converts to whatever dtype/layout its
runtime needs *without changing values or channel order*).

Output convention: softmax over the class logits. MiniFASNet variants are
either 2-class ``[fake, real]`` or 3-class ``[2D-spoof, real, 3D-spoof]``.
In BOTH layouts the *real* class is index 1, so ``softmax(logits)[1]`` is
the real-face probability.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

# Canonical MiniFASNet 2.7_80x80 input parameters.
LIVENESS_INPUT_SIZE = 80
LIVENESS_CROP_SCALE = 2.7

# Index of the "real" class in the softmax output (same for 2- and 3-class).
LIVENESS_REAL_INDEX = 1


def get_expanded_box(
    src_w: int,
    src_h: int,
    bbox_xywh: Tuple[float, float, float, float],
    scale: float = LIVENESS_CROP_SCALE,
) -> Tuple[int, int, int, int]:
    """Expand a detection bbox around its center by ``scale``.

    Faithful port of Minivision ``CropImage._get_new_box``: the effective
    scale is clamped so the expanded box fits inside the image; if the box
    crosses a border it is shifted back inside, then clamped.

    Args:
        src_w / src_h: source image dimensions.
        bbox_xywh: raw detection box ``(x, y, w, h)`` in image coordinates.
        scale: expansion factor (2.7 for MiniFASNet 2.7_80x80).

    Returns:
        Inclusive box corners ``(left, top, right, bottom)`` as ints,
        guaranteed inside ``[0, src_w-1] x [0, src_h-1]``.
    """
    x, y, box_w, box_h = bbox_xywh
    if box_w <= 0 or box_h <= 0:
        raise ValueError(f"Invalid bbox for liveness crop: w={box_w}, h={box_h}")

    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)

    new_w = box_w * scale
    new_h = box_h * scale
    center_x = x + box_w / 2
    center_y = y + box_h / 2

    left = center_x - new_w / 2
    top = center_y - new_h / 2
    right = center_x + new_w / 2
    bottom = center_y + new_h / 2

    # Shift back inside the image instead of truncating (official behavior).
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > src_w - 1:
        left -= right - (src_w - 1)
        right = src_w - 1
    if bottom > src_h - 1:
        top -= bottom - (src_h - 1)
        bottom = src_h - 1

    # Safety clamp (scale clamping above should already guarantee this).
    left = max(0.0, left)
    top = max(0.0, top)
    return int(left), int(top), int(right), int(bottom)


def crop_liveness_input(
    image_bgr: np.ndarray,
    bbox_xywh: Tuple[float, float, float, float],
    scale: float = LIVENESS_CROP_SCALE,
    out_size: int = LIVENESS_INPUT_SIZE,
) -> np.ndarray:
    """Crop + resize a face region into the canonical MiniFASNet input.

    Args:
        image_bgr: full original BGR uint8 image.
        bbox_xywh: raw detection bbox ``(x, y, w, h)``.
        scale: bbox expansion factor.
        out_size: output side length (80).

    Returns:
        ``(out_size, out_size, 3)`` BGR uint8 crop. Deliberately **no**
        normalization — see module docstring.
    """
    src_h, src_w = image_bgr.shape[:2]
    left, top, right, bottom = get_expanded_box(src_w, src_h, bbox_xywh, scale)
    crop = image_bgr[top : bottom + 1, left : right + 1]
    if crop.size == 0:
        raise ValueError(
            f"Empty liveness crop (bbox={bbox_xywh}, image={src_w}x{src_h})"
        )
    return cv2.resize(crop, (out_size, out_size))


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax over a 1-D logit vector."""
    logits = logits.astype(np.float64).flatten()
    e = np.exp(logits - np.max(logits))
    return (e / np.sum(e)).astype(np.float32)
