import cv2
import numpy as np


def order_points(pts):
    """
    Arrange four points as:
    top-left, top-right, bottom-right, bottom-left.
    """

    rect = np.zeros((4, 2), dtype="float32")

    # Sum of x and y coordinates
    s = pts.sum(axis=1)

    # Top-left has the smallest sum
    rect[0] = pts[np.argmin(s)]

    # Bottom-right has the largest sum
    rect[2] = pts[np.argmax(s)]

    # Difference between y and x
    diff = np.diff(pts, axis=1)

    # Top-right has the smallest difference
    rect[1] = pts[np.argmin(diff)]

    # Bottom-left has the largest difference
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """
    Transform the four document corners
    into a straight rectangular document.
    """

    # Make sure points are float32
    pts = pts.astype("float32")

    # Arrange points
    rect = order_points(pts)

    tl, tr, br, bl = rect

    # Calculate top and bottom widths
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)

    max_width = max(
        int(width_top),
        int(width_bottom)
    )

    # Calculate left and right heights
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    max_height = max(
        int(height_left),
        int(height_right)
    )

    # Prevent invalid output dimensions
    if max_width <= 0 or max_height <= 0:
        raise ValueError(
            "Invalid document dimensions."
        )

    # Destination rectangle
    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ],
        dtype="float32"
    )

    # Calculate transformation matrix
    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    # Apply transformation
    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height)
    )

    return warped
