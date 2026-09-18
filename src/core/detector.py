import cv2
import numpy as np
import os
import glob

from .transformer import four_point_transform


# ============================================================
# Project paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

SAMPLES_DIR = os.path.join(
    BASE_DIR,
    "samples"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Load image
# ============================================================

def load_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Image could not be loaded: {image_path}"
        )

    print("Image loaded successfully.")

    return image


# ============================================================
# Preprocessing
# ============================================================

def preprocess(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    print("Preprocessing completed.")

    return gray, blurred


# ============================================================
# Canny edge detection
# ============================================================

def detect_edges(blurred):

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    print("Canny edge detection completed.")

    return edges


# ============================================================
# Create document mask using GrabCut
# ============================================================

def create_document_mask(image):

    print("\nCreating document mask using GrabCut...")

    height, width = image.shape[:2]

    # Start with everything as background.
    mask = np.full(
        (height, width),
        cv2.GC_BGD,
        dtype=np.uint8
    )

    # Give a small border as definite background.
    border = 10

    mask[
        border:height - border,
        border:width - border
    ] = cv2.GC_PR_BGD

    # The central area is considered probable foreground.
    # This works well because the document occupies
    # the main central part of the photograph.
    x1 = int(width * 0.08)
    x2 = int(width * 0.90)

    y1 = int(height * 0.12)
    y2 = int(height * 0.80)

    mask[
        y1:y2,
        x1:x2
    ] = cv2.GC_PR_FGD

    # The center of the image is marked as sure foreground.
    # This gives GrabCut a strong starting point.
    center_x1 = int(width * 0.20)
    center_x2 = int(width * 0.80)

    center_y1 = int(height * 0.25)
    center_y2 = int(height * 0.75)

    mask[
        center_y1:center_y2,
        center_x1:center_x2
    ] = cv2.GC_FGD

    # GrabCut working arrays.
    background_model = np.zeros(
        (1, 65),
        np.float64
    )

    foreground_model = np.zeros(
        (1, 65),
        np.float64
    )

    # Run GrabCut.
    cv2.grabCut(
        image,
        mask,
        None,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_MASK
    )

    # Convert GrabCut output into a binary mask.
    document_mask = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # Close small gaps.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7)
    )

    document_mask = cv2.morphologyEx(
        document_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Remove very small noise.
    document_mask = cv2.morphologyEx(
        document_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    print("Document mask created.")

    return document_mask


# ============================================================
# Prepare edge mask
# ============================================================

def prepare_edges(edges):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    dilated = cv2.dilate(
        closed,
        kernel,
        iterations=1
    )

    return closed, dilated


# ============================================================
# Check whether contour touches image border
# ============================================================

def is_border_contour(
    contour,
    image_width,
    image_height
):

    x, y, width, height = cv2.boundingRect(
        contour
    )

    margin = 5

    if x <= margin:
        return True

    if y <= margin:
        return True

    if x + width >= image_width - margin:
        return True

    if y + height >= image_height - margin:
        return True

    return False


# ============================================================
# Find document contour
# ============================================================

def find_document_contour(
    document_mask,
    image
):

    print("\nSearching for document contour...")

    height, width = image.shape[:2]

    image_area = width * height

    minimum_area = image_area * 0.20
    maximum_area = image_area * 0.95

    contours, _ = cv2.findContours(
        document_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(
        "Contours detected:",
        len(contours)
    )

    if len(contours) == 0:

        print("\nDOCUMENT NOT FOUND")

        return None

    # Sort contours from largest to smallest.
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        print(
            "Checking contour area:",
            int(area)
        )

        # Ignore very small contours.
        if area < minimum_area:
            continue

        # Ignore almost full-image contours.
        if area > maximum_area:
            print(
                "Contour too large, ignored."
            )
            continue

        # Ignore the outer photograph border.
        if is_border_contour(
            contour,
            width,
            height
        ):

            print(
                "Border contour ignored."
            )

            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        # Try multiple approximation values.
        for epsilon_ratio in [
            0.005,
            0.01,
            0.015,
            0.02,
            0.025,
            0.03
        ]:

            approximation = cv2.approxPolyDP(
                contour,
                epsilon_ratio * perimeter,
                True
            )

            print(
                "Contour area:",
                int(area),
                "| Approximation:",
                epsilon_ratio,
                "| Corners:",
                len(approximation)
            )

            # Document must have four corners.
            if len(approximation) != 4:
                continue

            points = approximation.reshape(
                4,
                2
            ).astype(np.float32)

            # Find bounding dimensions.
            x_values = points[:, 0]
            y_values = points[:, 1]

            document_width = (
                np.max(x_values)
                - np.min(x_values)
            )

            document_height = (
                np.max(y_values)
                - np.min(y_values)
            )

            # Document should be reasonably large.
            if document_width < width * 0.40:
                continue

            if document_height < height * 0.40:
                continue

            # Make sure points are not on photograph edges.
            if np.any(points[:, 0] <= 3):
                continue

            if np.any(points[:, 1] <= 3):
                continue

            if np.any(
                points[:, 0] >= width - 3
            ):
                continue

            if np.any(
                points[:, 1] >= height - 3
            ):
                continue

            # Check that the contour is convex.
            if not cv2.isContourConvex(
                approximation
            ):
                continue

            print(
                "\nDOCUMENT FOUND"
            )

            return points

    print(
        "\nDOCUMENT NOT FOUND"
    )

    return None


# ============================================================
# Draw detected contour
# ============================================================

def draw_contour(
    image,
    corners
):

    result = image.copy()

    if corners is None:

        return result

    points = corners.astype(
        np.int32
    )

    # Draw document boundary.
    cv2.polylines(
        result,
        [points],
        True,
        (0, 255, 0),
        4
    )

    # Draw four corner points.
    for i, point in enumerate(points):

        x = int(point[0])
        y = int(point[1])

        cv2.circle(
            result,
            (x, y),
            8,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            result,
            f"P{i + 1}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    return result


# ============================================================
# Save image
# ============================================================

def save_image(
    image,
    path
):

    success = cv2.imwrite(
        path,
        image
    )

    print(
        "Saved:",
        os.path.basename(path),
        "| Success:",
        success
    )

    return success


# ============================================================
# Process one image
# ============================================================

def process_image(image_path):

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    print("\n" + "=" * 60)

    print(
        "Processing:",
        image_name
    )

    print(
        "Input path:",
        image_path
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Step 1: Load image
    # --------------------------------------------------------

    image = load_image(
        image_path
    )

    # --------------------------------------------------------
    # Step 2: Preprocessing
    # --------------------------------------------------------

    gray, blurred = preprocess(
        image
    )

    # --------------------------------------------------------
    # Step 3: Edge detection
    # --------------------------------------------------------

    edges = detect_edges(
        blurred
    )

    # --------------------------------------------------------
    # Step 4: Create document mask
    # --------------------------------------------------------

    document_mask = create_document_mask(
        image
    )

    # --------------------------------------------------------
    # Step 5: Prepare edges
    # --------------------------------------------------------

    closed_edges, dilated_edges = prepare_edges(
        edges
    )

    # --------------------------------------------------------
    # Step 6: Find document contour
    # --------------------------------------------------------

    corners = find_document_contour(
        document_mask,
        image
    )

    # --------------------------------------------------------
    # Step 7: Save preprocessing results
    # --------------------------------------------------------

    save_image(
        gray,
        os.path.join(
            OUTPUT_DIR,
            image_name + "_gray.jpg"
        )
    )

    save_image(
        blurred,
        os.path.join(
            OUTPUT_DIR,
            image_name + "_blurred.jpg"
        )
    )

    save_image(
        edges,
        os.path.join(
            OUTPUT_DIR,
            image_name + "_edges.jpg"
        )
    )

    save_image(
        closed_edges,
        os.path.join(
            OUTPUT_DIR,
            image_name + "_closed.jpg"
        )
    )

    save_image(
        dilated_edges,
        os.path.join(
            OUTPUT_DIR,
            image_name + "_dilated.jpg"
        )
    )

    save_image(
        document_mask,
        os.path.join(
            OUTPUT_DIR,
            image_name + "_mask.jpg"
        )
    )

    # --------------------------------------------------------
    # Step 8: Save contour image
    # --------------------------------------------------------

    contour_image = draw_contour(
        image,
        corners
    )

    save_image(
        contour_image,
        os.path.join(
            OUTPUT_DIR,
            image_name + "_contour.jpg"
        )
    )

    # --------------------------------------------------------
    # Step 9: Perspective transformation
    # --------------------------------------------------------

    if corners is not None:

        print(
            "\nDetected document corners:"
        )

        for i, point in enumerate(corners):

            print(
                f"Corner {i + 1}: "
                f"({int(point[0])}, "
                f"{int(point[1])})"
            )

        try:

            warped = four_point_transform(
                image,
                corners
            )

            warped_path = os.path.join(
                OUTPUT_DIR,
                image_name + "_warped.jpg"
            )

            save_image(
                warped,
                warped_path
            )

            print(
                "\nPerspective transformation completed."
            )

            print(
                "Warped output:",
                warped_path
            )

        except Exception as error:

            print(
                "\nPerspective transformation failed."
            )

            print(
                "Error:",
                error
            )

    else:

        print(
            "\nPerspective transformation skipped."
        )

        print(
            "Reason: document contour "
            "was not detected."
        )


# ============================================================
# Find all input images
# ============================================================

def get_input_images():

    image_paths = []

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png"
    ]

    for extension in extensions:

        paths = glob.glob(
            os.path.join(
                SAMPLES_DIR,
                extension
            )
        )

        image_paths.extend(
            paths
        )

    return sorted(
        image_paths
    )


# ============================================================
# Main program
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "DOCUMENT SCANNER"
    )

    print(
        "MODULE 1 + MODULE 2"
    )

    print(
        "PREPROCESSING, EDGE DETECTION, "
        "DOCUMENT DETECTION AND "
        "PERSPECTIVE TRANSFORMATION"
    )

    print("=" * 60)

    # Find all images inside samples folder.
    image_paths = get_input_images()

    print(
        "\nImages found:"
    )

    if len(image_paths) == 0:

        print(
            "No images found."
        )

    else:

        for path in image_paths:

            print(
                "-",
                os.path.basename(path)
            )

        # Process every image.
        for image_path in image_paths:

            try:

                process_image(
                    image_path
                )

            except Exception as error:

                print(
                    "\nError while processing:"
                )

                print(
                    image_path
                )

                print(
                    "Error:",
                    error
                )

    print(
        "\n" + "=" * 60
    )

    print(
        "ALL PROCESSING COMPLETED"
    )

    print(
        "=" * 60
    )
