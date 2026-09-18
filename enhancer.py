import cv2
import numpy as np
import os
import glob


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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Sharpen image
# ============================================================

def sharpen(image):

    """
    Sharpen the image while keeping the text readable.
    """

    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        dtype=np.float32
    )

    return cv2.filter2D(
        image,
        -1,
        kernel
    )


# ============================================================
# Enhance document
# ============================================================

def enhance_document(warped):

    """
    Takes a perspective-corrected document and creates:

    1. Enhanced color document
    2. Black-and-white scanned document
    """

    if warped is None:

        raise ValueError(
            "Input warped image is empty."
        )

    # --------------------------------------------------------
    # Step 1: Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        warped,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Step 2: Improve local contrast
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    enhanced_gray = clahe.apply(
        gray
    )

    # --------------------------------------------------------
    # Step 3: Gentle sharpening
    # --------------------------------------------------------

    sharpen_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        dtype=np.float32
    )

    sharpened = cv2.filter2D(
        enhanced_gray,
        -1,
        sharpen_kernel
    )

    # --------------------------------------------------------
    # Step 4: Reduce small noise
    # --------------------------------------------------------

    denoised = cv2.GaussianBlur(
        sharpened,
        (3, 3),
        0
    )

    # --------------------------------------------------------
    # Step 5: Adaptive threshold
    # --------------------------------------------------------
    #
    # A larger neighbourhood is used because the notebook
    # page has uneven lighting and shadows.
    #
    # blockSize = 31
    # C = 7
    #
    # This is gentler than the previous processing and
    # helps preserve handwritten characters.
    # --------------------------------------------------------

    scanned_bw = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7
    )

    # --------------------------------------------------------
    # Step 6: Create enhanced color version
    # --------------------------------------------------------

    enhanced_color = cv2.filter2D(
        warped,
        -1,
        sharpen_kernel
    )

    return enhanced_color, scanned_bw


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
# Process one warped document
# ============================================================

def process_document(
    warped_path
):

    file_name = os.path.basename(
        warped_path
    )

    name = os.path.splitext(
        file_name
    )[0]

    print("\n" + "=" * 60)

    print(
        "Processing:",
        file_name
    )

    print(
        "Input:",
        warped_path
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Step 1: Load warped image
    # --------------------------------------------------------

    warped = cv2.imread(
        warped_path
    )

    if warped is None:

        print(
            "ERROR: Image could not be loaded."
        )

        return

    print(
        "Warped image loaded successfully."
    )

    # --------------------------------------------------------
    # Step 2: Enhance document
    # --------------------------------------------------------

    enhanced_color, scanned_bw = enhance_document(
        warped
    )

    print(
        "Document enhancement completed."
    )

    # --------------------------------------------------------
    # Step 3: Save enhanced color image
    # --------------------------------------------------------

    enhanced_color_path = os.path.join(
        OUTPUT_DIR,
        name + "_enhanced_color.jpg"
    )

    save_image(
        enhanced_color,
        enhanced_color_path
    )

    # --------------------------------------------------------
    # Step 4: Save black-and-white scanned image
    # --------------------------------------------------------

    scanned_bw_path = os.path.join(
        OUTPUT_DIR,
        name + "_scanned_bw.jpg"
    )

    save_image(
        scanned_bw,
        scanned_bw_path
    )

    print(
        "\nEnhancement completed successfully."
    )


# ============================================================
# Find warped images
# ============================================================

def get_warped_images():

    pattern = os.path.join(
        OUTPUT_DIR,
        "*_warped.jpg"
    )

    warped_images = glob.glob(
        pattern
    )

    return sorted(
        warped_images
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
        "MODULE 3"
    )

    print(
        "DOCUMENT ENHANCEMENT"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Find all warped documents
    # --------------------------------------------------------

    warped_images = get_warped_images()

    print(
        "\nWarped images found:"
    )

    if len(warped_images) == 0:

        print(
            "No warped images found."
        )

        print(
            "Please run detector.py first."
        )

    else:

        # Display all warped images found.
        for path in warped_images:

            print(
                "-",
                os.path.basename(path)
            )

        # ----------------------------------------------------
        # Process every warped document
        # ----------------------------------------------------

        for warped_path in warped_images:

            try:

                process_document(
                    warped_path
                )

            except Exception as error:

                print(
                    "\nError while processing:"
                )

                print(
                    warped_path
                )

                print(
                    "Error:",
                    error
                )

    print(
        "\n" + "=" * 60
    )

    print(
        "ALL ENHANCEMENT COMPLETED"
    )

    print(
        "=" * 60
    )