import os
import argparse


# ============================================================
# Project paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
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
# Import project modules
# ============================================================

try:

    from src.core.detector import (
        load_image,
        preprocess,
        detect_edges,
        create_document_mask,
        find_document_contour,
        draw_contour,
        save_image,
        get_input_images
    )

    from src.core.transformer import (
        four_point_transform
    )

    from src.core.enhancer import (
        enhance_document
    )

except ImportError as error:

    print(
        "ERROR: Could not import project modules."
    )

    print(
        "Details:",
        error
    )

    print(
        "\nCheck that your project structure is:"
    )

    print(
        "src/"
    )

    print(
        "  __init__.py"
    )

    print(
        "  core/"
    )

    print(
        "    __init__.py"
    )

    print(
        "    detector.py"
    )

    print(
        "    transformer.py"
    )

    print(
        "    enhancer.py"
    )

    raise SystemExit(1)


# ============================================================
# Scan one document
# ============================================================

def scan_document(
    image_path,
    output_dir=OUTPUT_DIR
):

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not os.path.exists(
        image_path
    ):

        print(
            "\nERROR: Image does not exist."
        )

        print(
            "Path:",
            image_path
        )

        return None

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Get image name
    # --------------------------------------------------------

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    print(
        "\n" + "=" * 60
    )

    print(
        "Scanning:",
        image_name
    )

    print(
        "Input:",
        image_path
    )

    print(
        "=" * 60
    )

    try:

        # ====================================================
        # Module 1: Load image
        # ====================================================

        image = load_image(
            image_path
        )

        # ====================================================
        # Module 1: Preprocessing
        # ====================================================

        gray, blurred = preprocess(
            image
        )

        save_image(
            gray,
            os.path.join(
                output_dir,
                image_name + "_gray.jpg"
            )
        )

        save_image(
            blurred,
            os.path.join(
                output_dir,
                image_name + "_blurred.jpg"
            )
        )

        # ====================================================
        # Module 1: Canny edge detection
        # ====================================================

        edges = detect_edges(
            blurred
        )

        save_image(
            edges,
            os.path.join(
                output_dir,
                image_name + "_edges.jpg"
            )
        )

        # ====================================================
        # Document mask
        # ====================================================

        document_mask = create_document_mask(
            image
        )

        save_image(
            document_mask,
            os.path.join(
                output_dir,
                image_name + "_mask.jpg"
            )
        )

        # ====================================================
        # Document contour detection
        # ====================================================

        corners = find_document_contour(
            document_mask,
            image
        )

        contour_image = draw_contour(
            image,
            corners
        )

        contour_path = os.path.join(
            output_dir,
            image_name + "_contour.jpg"
        )

        save_image(
            contour_image,
            contour_path
        )

        # ====================================================
        # Check document detection
        # ====================================================

        if corners is None:

            print(
                "\nDOCUMENT NOT DETECTED."
            )

            print(
                "Perspective transformation skipped."
            )

            return None

        # ====================================================
        # Display detected corners
        # ====================================================

        print(
            "\nDetected document corners:"
        )

        for i, point in enumerate(
            corners
        ):

            print(
                f"Corner {i + 1}: "
                f"({int(point[0])}, "
                f"{int(point[1])})"
            )

        # ====================================================
        # Module 2: Perspective transformation
        # ====================================================

        warped = four_point_transform(
            image,
            corners
        )

        warped_path = os.path.join(
            output_dir,
            image_name + "_warped.jpg"
        )

        save_image(
            warped,
            warped_path
        )

        print(
            "\nPerspective transformation completed."
        )

        # ====================================================
        # Module 3: Enhancement
        # ====================================================

        # Your current enhancer.py returns 2 values.
        enhanced_color, scanned_bw = enhance_document(
            warped
        )

        print(
            "Document enhancement completed."
        )

        # ====================================================
        # Save enhanced image
        # ====================================================

        enhanced_path = os.path.join(
            output_dir,
            image_name + "_enhanced.jpg"
        )

        save_image(
            enhanced_color,
            enhanced_path
        )

        # ====================================================
        # Save final scanned image
        # ====================================================

        scanned_path = os.path.join(
            output_dir,
            image_name + "_scanned.jpg"
        )

        save_image(
            scanned_bw,
            scanned_path
        )

        # ====================================================
        # Final output information
        # ====================================================

        print(
            "\nDocument scanning completed successfully."
        )

        print(
            "Contour:",
            contour_path
        )

        print(
            "Warped:",
            warped_path
        )

        print(
            "Enhanced:",
            enhanced_path
        )

        print(
            "Final scan:",
            scanned_path
        )

        print(
            "\nDone. Final scan:",
            image_name + "_scanned.jpg"
        )

        print(
            "=" * 60
        )

        return scanned_bw

    except Exception as error:

        print(
            "\nERROR while processing:"
        )

        print(
            "Image:",
            image_path
        )

        print(
            "Error:",
            error
        )

        return None


# ============================================================
# Convert path to absolute path
# ============================================================

def get_absolute_image_path(
    image_path
):

    if os.path.isabs(
        image_path
    ):

        return os.path.normpath(
            image_path
        )

    # Try path relative to project folder.
    project_path = os.path.join(
        BASE_DIR,
        image_path
    )

    return os.path.normpath(
        project_path
    )


# ============================================================
# Scan all sample images
# ============================================================

def scan_all():

    print(
        "\nSearching for images in samples folder..."
    )

    image_paths = get_input_images()

    if len(image_paths) == 0:

        print(
            "\nNo images found in:"
        )

        print(
            SAMPLES_DIR
        )

        return

    print(
        "\nImages found:"
    )

    for path in image_paths:

        print(
            "-",
            os.path.basename(path)
        )

    for image_path in image_paths:

        scan_document(
            image_path
        )


# ============================================================
# Main function
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Document Scanner - detects, "
            "flattens and enhances photographed documents."
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Path to a single image to scan."
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every image in samples/."
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Process one image
    # --------------------------------------------------------

    if args.image:

        image_path = get_absolute_image_path(
            args.image
        )

        scan_document(
            image_path
        )

    # --------------------------------------------------------
    # Process all images
    # --------------------------------------------------------

    elif args.all:

        scan_all()

    # --------------------------------------------------------
    # No arguments
    # --------------------------------------------------------

    else:

        parser.print_help()

        print(
            "\nExamples:"
        )

        print(
            "python main.py --all"
        )

        print(
            "python main.py --image samples\\test1.jpg"
        )


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":

    main()