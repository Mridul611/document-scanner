import numpy as np

from src.core.enhancer import sharpen


# ============================================================
# Test image shape
# ============================================================

def test_sharpen_preserves_shape():

    dummy = np.zeros(
        (50, 50, 3),
        dtype=np.uint8
    )

    result = sharpen(
        dummy
    )

    assert result.shape == dummy.shape


# ============================================================
# Test output data type
# ============================================================

def test_sharpen_preserves_dtype():

    dummy = np.zeros(
        (50, 50, 3),
        dtype=np.uint8
    )

    result = sharpen(
        dummy
    )

    assert result.dtype == np.uint8


# ============================================================
# Test that sharpening affects image data
# ============================================================

def test_sharpen_changes_image():

    # Create a deterministic image with different
    # intensity values so that sharpening has an
    # actual variation to process.
    rng = np.random.default_rng(42)

    dummy = rng.integers(
        50,
        200,
        size=(50, 50, 3),
        dtype=np.uint8
    )

    result = sharpen(
        dummy
    )

    # Check that the sharpening operation changes
    # at least some pixel values.
    difference = np.abs(
        result.astype(np.int16)
        - dummy.astype(np.int16)
    )

    assert np.any(
        difference > 0
    )


# ============================================================
# Test valid pixel values
# ============================================================

def test_sharpen_valid_pixel_values():

    dummy = np.full(
        (50, 50, 3),
        120,
        dtype=np.uint8
    )

    dummy[10:40, 10:40] = 200

    result = sharpen(
        dummy
    )

    assert np.min(result) >= 0
    assert np.max(result) <= 255
