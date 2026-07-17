"""Numerical regression tests for the pure-numpy Umeyama similarity transform.

The reference implementation is ``skimage.transform.SimilarityTransform``,
which ``umeyama_similarity`` replaced to drop the scikit-image runtime
dependency. The comparison tests only run when scikit-image is importable
(install it in the *test* environment only, e.g.
``uv run --with scikit-image pytest tests/test_umeyama.py``); the
self-consistency tests always run.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from face_pipeline import ARCFACE_DEST_LANDMARKS, umeyama_similarity  # noqa: E402

skimage_transform = pytest.importorskip(
    "skimage.transform", reason="scikit-image not installed (test-env-only dep)"
)


def _skimage_matrix_raw(src, dst):
    tform = skimage_transform.SimilarityTransform()
    tform.estimate(src, dst)
    return np.asarray(tform.params)


def _skimage_matrix(src, dst):
    return _skimage_matrix_raw(np.asarray(src, dtype=np.float64),
                               np.asarray(dst, dtype=np.float64))


def _assert_matches_skimage(src, dst, atol=1e-6):
    # Feed bit-identical float64 inputs to both implementations —
    # umeyama_similarity intentionally computes in the input dtype
    # (see its docstring), so dtype must not differ between the two.
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    ours = umeyama_similarity(src, dst)
    ref = _skimage_matrix(src, dst)
    np.testing.assert_allclose(ours, ref, atol=atol, rtol=0)


class TestVsSkimage:
    def test_random_point_sets(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            n = int(rng.integers(3, 12))
            src = rng.uniform(-100, 100, size=(n, 2))
            dst = rng.uniform(-100, 100, size=(n, 2))
            _assert_matches_skimage(src, dst)

    def test_known_transform_recovered(self):
        """Points related by an exact similarity transform: matrix must
        reproduce it (and match skimage)."""
        rng = np.random.default_rng(7)
        src = rng.uniform(0, 50, size=(5, 2))
        theta = 0.3
        s = 1.7
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        t = np.array([12.5, -3.25])
        dst = (s * (src @ R.T)) + t
        T = umeyama_similarity(src, dst)
        expected = np.eye(3)
        expected[:2, :2] = s * R
        expected[:2, 2] = t
        np.testing.assert_allclose(T, expected, atol=1e-9)
        _assert_matches_skimage(src, dst)

    def test_reflection_case(self):
        """dst is a mirrored copy of src -> det(cov) < 0 path (d correction)."""
        rng = np.random.default_rng(3)
        src = rng.uniform(-10, 10, size=(6, 2))
        dst = src * np.array([-1.0, 1.0]) + np.array([4.0, -2.0])
        _assert_matches_skimage(src, dst)

    def test_arcface_landmarks(self):
        """The actual production use: 5 noisy landmarks -> ArcFace canon."""
        rng = np.random.default_rng(11)
        for _ in range(20):
            jitter = rng.normal(0, 8.0, size=(5, 2))
            shift = rng.uniform(-200, 200, size=(1, 2))
            scale = rng.uniform(0.3, 4.0)
            src = ARCFACE_DEST_LANDMARKS * scale + jitter + shift
            _assert_matches_skimage(src, ARCFACE_DEST_LANDMARKS)

    def test_float32_bit_parity(self):
        """Production path: float32 landmarks. Must be BIT-identical to
        skimage (int8 NPU embedders amplify 1-LSB warp differences)."""
        rng = np.random.default_rng(23)
        for _ in range(20):
            src = (ARCFACE_DEST_LANDMARKS
                   + rng.normal(0, 10, size=(5, 2))).astype(np.float32)
            ours = umeyama_similarity(src, ARCFACE_DEST_LANDMARKS)
            ref = _skimage_matrix_raw(src, ARCFACE_DEST_LANDMARKS)
            assert np.array_equal(ours, ref)

    def test_collinear_points(self):
        """Degenerate rank-deficient input behaves like skimage."""
        src = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        dst = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
        ours = umeyama_similarity(src, dst)
        ref = _skimage_matrix(src, dst)
        if np.isnan(ref).any():
            assert np.isnan(ours).any()
        else:
            np.testing.assert_allclose(ours, ref, atol=1e-6, rtol=0)

    def test_all_identical_points(self):
        """Rank-0 covariance -> NaN matrix, same as skimage."""
        src = np.ones((5, 2)) * 3.0
        dst = np.ones((5, 2)) * 7.0
        ours = umeyama_similarity(src, dst)
        ref = _skimage_matrix(src, dst)
        assert np.isnan(ours).all() == np.isnan(ref).all()
        assert np.isnan(ours).any()


class TestSelfConsistency:
    """Always-on tests (no skimage needed beyond module import above)."""

    def test_similarity_structure(self):
        """Result is a proper similarity: R orthogonal * uniform scale."""
        rng = np.random.default_rng(5)
        src = rng.uniform(0, 100, size=(5, 2))
        dst = rng.uniform(0, 100, size=(5, 2))
        T = umeyama_similarity(src, dst)
        A = T[:2, :2]
        s = np.sqrt(np.abs(np.linalg.det(A)))
        R = A / s
        np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-9)
        assert T[2, 0] == 0 and T[2, 1] == 0 and T[2, 2] == 1
