"""Unit tests for the shared MiniFASNet crop/preprocess util (liveness.py)."""
import numpy as np
import pytest

from liveness import (
    LIVENESS_INPUT_SIZE,
    LIVENESS_REAL_INDEX,
    crop_liveness_input,
    get_expanded_box,
    softmax,
)


def test_interior_box_expansion():
    # bbox (50,50,20,20) in 200x200, scale 2.7 -> 54x54 centered at (60,60).
    left, top, right, bottom = get_expanded_box(200, 200, (50, 50, 20, 20), 2.7)
    assert (left, top, right, bottom) == (33, 33, 87, 87)


def test_border_box_is_shifted_inside():
    # bbox at the corner: expanded box would go negative; official CropImage
    # shifts it back inside rather than truncating.
    left, top, right, bottom = get_expanded_box(200, 200, (0, 0, 20, 20), 2.7)
    assert (left, top) == (0, 0)
    assert (right, bottom) == (54, 54)  # full 54-px extent preserved by shift


def test_large_box_scale_is_clamped():
    # Expanded box can never exceed the source image.
    left, top, right, bottom = get_expanded_box(200, 200, (0, 0, 150, 150), 2.7)
    assert 0 <= left <= right <= 199
    assert 0 <= top <= bottom <= 199


def test_box_never_exceeds_image_bounds_fuzz():
    rng = np.random.default_rng(42)
    for _ in range(200):
        src_w = int(rng.integers(50, 800))
        src_h = int(rng.integers(50, 800))
        w = int(rng.integers(1, src_w))
        h = int(rng.integers(1, src_h))
        x = int(rng.integers(-10, src_w))
        y = int(rng.integers(-10, src_h))
        left, top, right, bottom = get_expanded_box(src_w, src_h, (x, y, w, h))
        assert 0 <= left <= right <= src_w - 1
        assert 0 <= top <= bottom <= src_h - 1


def test_degenerate_bbox_raises():
    with pytest.raises(ValueError):
        get_expanded_box(200, 200, (10, 10, 0, 20))
    with pytest.raises(ValueError):
        get_expanded_box(200, 200, (10, 10, 20, -1))


def test_crop_output_shape_and_dtype():
    img = np.random.default_rng(0).integers(
        0, 255, size=(240, 320, 3), dtype=np.uint8
    ).astype(np.uint8)
    crop = crop_liveness_input(img, (100, 80, 40, 50))
    assert crop.shape == (LIVENESS_INPUT_SIZE, LIVENESS_INPUT_SIZE, 3)
    assert crop.dtype == np.uint8


def test_crop_no_normalization_values_preserved():
    # A constant-value image must stay constant after crop+resize — i.e. no
    # mean/std normalization or /255 scaling happens in the shared util.
    img = np.full((200, 200, 3), 200, dtype=np.uint8)
    crop = crop_liveness_input(img, (50, 50, 40, 40))
    assert crop.min() == crop.max() == 200


def test_softmax_real_index():
    probs = softmax(np.array([0.0, 3.0, 0.0], dtype=np.float32))
    assert probs.shape == (3,)
    assert abs(float(probs.sum()) - 1.0) < 1e-6
    assert LIVENESS_REAL_INDEX == 1
    assert probs[1] > probs[0] and probs[1] > probs[2]
