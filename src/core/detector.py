import cv2
import os
import glob


# find the main project folder
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

# input and output folders
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image(image_path):
    # load the image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Image could not be loaded: {image_path}"
        )

    print("Image loaded successfully.")

    return image


def preprocess(image):
    # convert image to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # reduce image noise
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    print("Preprocessing completed.")

    return gray, blurred


def detect_edges(blurred):
    # detect edges using Canny
    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    print("Canny edge detection completed.")

    return edges


def create_document_mask(gray):
    # create a binary image using adaptive threshold
    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # reverse the image
    threshold = cv2.bitwise_not(threshold)

    # connect nearby regions
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7)
    )

    mask = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return mask


def prepare_contours(edges, gray):
    print("\nPreparing contours...")

    # close gaps in Canny edges
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    closed_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # make nearby edges connected
    dilated_edges = cv2.dilate(
        closed_edges,
        kernel,
        iterations=1
    )

    # create another boundary image
    document_mask = create_document_mask(
        gray
    )

    # combine the two boundary images
    combined = cv2.bitwise_or(
        dilated_edges,
        document_mask
    )

    return (
        closed_edges,
        dilated_edges,
        document_mask,
        combined
    )


def get_contours(binary_image):
    # find only external contours
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # sort from largest to smallest
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    return contours


def find_four_corner_contour(
    binary_image,
    image_shape
):
    print("\nSearching for document contour...")

    contours = get_contours(
        binary_image
    )

    print(
        "Contours detected:",
        len(contours)
    )

    image_height = image_shape[0]
    image_width = image_shape[1]

    image_area = image_height * image_width

    # ignore very small contours
    minimum_area = image_area * 0.05

    checked_contours = 0

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < minimum_area:
            continue

        checked_contours += 1

        perimeter = cv2.arcLength(
            contour,
            True
        )

        # try different approximation values
        approximation_values = [
            0.005,
            0.01,
            0.015,
            0.02,
            0.025,
            0.03
        ]

        for epsilon_ratio in approximation_values:

            approx = cv2.approxPolyDP(
                contour,
                epsilon_ratio * perimeter,
                True
            )

            corners = len(approx)

            print(
                "Contour area:",
                int(area),
                "| Approximation:",
                epsilon_ratio,
                "| Corners:",
                corners
            )

            if corners == 4:

                # check that the four points make a reasonable shape
                points = approx.reshape(
                    4,
                    2
                )

                x_values = points[:, 0]
                y_values = points[:, 1]

                width = (
                    max(x_values)
                    - min(x_values)
                )

                height = (
                    max(y_values)
                    - min(y_values)
                )

                # document should have reasonable width and height
                if width > image_width * 0.25 and \
                   height > image_height * 0.25:

                    print(
                        "\nDOCUMENT FOUND"
                    )

                    print(
                        "Four corner points detected."
                    )

                    return points

    print(
        "\nNo suitable four-corner contour found."
    )

    print(
        "Large contours checked:",
        checked_contours
    )

    return None


def draw_document_contour(
    image,
    corners
):
    # make a copy of original image
    result = image.copy()

    if corners is None:
        return result

    # convert points to integer values
    points = corners.astype(
        "int32"
    )

    # draw the document boundary
    cv2.polylines(
        result,
        [points],
        True,
        (0, 255, 0),
        4
    )

    # draw each corner
    for i, point in enumerate(points):

        x = int(point[0])
        y = int(point[1])

        cv2.circle(
            result,
            (x, y),
            10,
            (0, 0, 255),
            -1
        )

        # write corner number
        cv2.putText(
            result,
            f"P{i + 1}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    return result


def save_image(
    image,
    path
):
    # save processed image
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


def process_image(image_path):

    print(
        "\n# ============================================================"
    )

    print(
        "Processing:",
        os.path.basename(image_path)
    )

    print(
        "# ============================================================"
    )

    print(
        "\nInput path:",
        image_path
    )

    # load image
    image = load_image(
        image_path
    )

    # preprocessing
    gray, blurred = preprocess(
        image
    )

    # Canny edge detection
    edges = detect_edges(
        blurred
    )

    # prepare different contour images
    (
        closed_edges,
        dilated_edges,
        document_mask,
        combined
    ) = prepare_contours(
        edges,
        gray
    )

    # try to find the document
    corners = find_four_corner_contour(
        combined,
        image.shape
    )

    # get image name
    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    # create output paths
    gray_path = os.path.join(
        OUTPUT_DIR,
        image_name + "_gray.jpg"
    )

    blurred_path = os.path.join(
        OUTPUT_DIR,
        image_name + "_blurred.jpg"
    )

    edges_path = os.path.join(
        OUTPUT_DIR,
        image_name + "_edges.jpg"
    )

    closed_path = os.path.join(
        OUTPUT_DIR,
        image_name + "_closed.jpg"
    )

    dilated_path = os.path.join(
        OUTPUT_DIR,
        image_name + "_dilated.jpg"
    )

    mask_path = os.path.join(
        OUTPUT_DIR,
        image_name + "_mask.jpg"
    )

    combined_path = os.path.join(
        OUTPUT_DIR,
        image_name + "_combined.jpg"
    )

    contour_path = os.path.join(
        OUTPUT_DIR,
        image_name + "_contour.jpg"
    )

    # save preprocessing results
    save_image(
        gray,
        gray_path
    )

    save_image(
        blurred,
        blurred_path
    )

    save_image(
        edges,
        edges_path
    )

    save_image(
        closed_edges,
        closed_path
    )

    save_image(
        dilated_edges,
        dilated_path
    )

    save_image(
        document_mask,
        mask_path
    )

    save_image(
        combined,
        combined_path
    )

    # draw detected document
    contour_image = draw_document_contour(
        image,
        corners
    )

    save_image(
        contour_image,
        contour_path
    )

    # display corner coordinates
    if corners is not None:

        print(
            "\nDocument corner points:"
        )

        for i, point in enumerate(
            corners
        ):

            print(
                f"Corner {i + 1}: "
                f"({int(point[0])}, "
                f"{int(point[1])})"
            )

    else:

        print(
            "\nDOCUMENT NOT FOUND"
        )

        print(
            "No four-corner contour was detected."
        )

    print(
        "\nOutput files:"
    )

    print(
        gray_path
    )

    print(
        blurred_path
    )

    print(
        edges_path
    )

    print(
        closed_path
    )

    print(
        dilated_path
    )

    print(
        mask_path
    )

    print(
        combined_path
    )

    print(
        contour_path
    )


def get_input_images():

    # find all supported image files
    image_paths = []

    image_paths.extend(
        glob.glob(
            os.path.join(
                SAMPLES_DIR,
                "*.jpg"
            )
        )
    )

    image_paths.extend(
        glob.glob(
            os.path.join(
                SAMPLES_DIR,
                "*.jpeg"
            )
        )
    )

    image_paths.extend(
        glob.glob(
            os.path.join(
                SAMPLES_DIR,
                "*.png"
            )
        )
    )

    return sorted(
        image_paths
    )


if __name__ == "__main__":

    print(
        "# ============================================================"
    )

    print(
        "DOCUMENT SCANNER  MODULE 1 - "
        "EDGE AND CONTOUR DETECTION"
    )

    print(
        "# ============================================================"
    )

    # get all input images
    image_paths = get_input_images()

    print(
        "\nImages found:",
        [
            os.path.basename(path)
            for path in image_paths
        ]
    )

    if len(image_paths) == 0:

        print(
            "\nNo images found."
        )

        print(
            "Add images inside:"
        )

        print(
            SAMPLES_DIR
        )

    else:

        # process every input image
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
        "\n# ============================================================"
    )

    print(
        "ALL PROCESSING COMPLETED"
    )

    print(
        "# ============================================================"
    )