import os
import sys

import numpy as np


# Add project root to Python path
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(
        0,
        BASE_DIR
    )


from src.core.transformer import order_points


# ============================================================
# Test basic point ordering
# ============================================================

def test_order_points_basic():

    pts = np.array(
        [
            [50, 200],   # Bottom-left
            [10, 10],    # Top-left
            [220, 210],  # Bottom-right
            [200, 10],   # Top-right
        ],
        dtype="float32"
    )

    result = order_points(
        pts
    )

    # Check shape
    assert result.shape == (4, 2)

    # Extract ordered points
    tl, tr, br, bl = result

    # Check exact expected order
    np.testing.assert_array_equal(
        tl,
        [10, 10]
    )

    np.testing.assert_array_equal(
        tr,
        [200, 10]
    )

    np.testing.assert_array_equal(
        br,
        [220, 210]
    )

    np.testing.assert_array_equal(
        bl,
        [50, 200]
    )


# ============================================================
# Test that returned points are float32
# ============================================================

def test_order_points_dtype():

    pts = np.array(
        [
            [220, 210],
            [10, 10],
            [50, 200],
            [200, 10],
        ],
        dtype="float32"
    )

    result = order_points(
        pts
    )

    assert result.dtype == np.float32


# ============================================================
# Test with shuffled input points
# ============================================================

def test_order_points_shuffled():

    pts = np.array(
        [
            [200, 10],
            [50, 200],
            [10, 10],
            [220, 210],
        ],
        dtype="float32"
    )

    result = order_points(
        pts
    )

    expected = np.array(
        [
            [10, 10],    # Top-left
            [200, 10],   # Top-right
            [220, 210],  # Bottom-right
            [50, 200],   # Bottom-left
        ],
        dtype="float32"
    )

    np.testing.assert_array_equal(
        result,
        expected
    )
