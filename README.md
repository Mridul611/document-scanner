# Document Scanner

A command-line document scanner that takes a photograph of a paper document, detects the document boundary, corrects its perspective, and produces an enhanced scanned-looking output.

## Features

- Detects the document region from a photograph using GrabCut-based segmentation and contour detection
- Uses Canny edge detection as part of the image preprocessing stage
- Automatically detects and identifies the four document corners
- Orders the detected corners correctly
- Applies perspective transformation to flatten a tilted document into a rectangular view
- Enhances the transformed document using CLAHE-based contrast enhancement
- Uses sharpening to improve text and edge clarity
- Applies Gaussian blur to reduce small noise
- Uses adaptive thresholding to create a black-and-white scanned version
- Supports scanning a single image
- Supports processing all sample images in batch mode
- Fully command-line based with no GUI required

## Technologies

- Python 3.10+
- OpenCV
- NumPy
- Pytest

## Project Structure

```text
document-scanner/
├── main.py                    # CLI entry point
├── src/
│   ├── __init__.py
│   └── core/
│       ├── __init__.py
│       ├── detector.py        # Document detection and contour detection
│       ├── transformer.py     # Perspective transformation
│       └── enhancer.py        # Document enhancement and scanning
├── samples/                   # Input test images
│   ├── test1.jpg
│   └── test2.jpg
├── outputs/                   # Generated processing results
├── docs/                      # Architecture and UML diagrams
├── tests/                     # Unit tests
│   ├── test_transformer.py
│   └── test_enhancer.py
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md
```



## How It Works

```text
Input Image
    ↓
Preprocessing
    ↓
Canny Edge Detection + GrabCut Mask
    ↓
Contour Detection
    ↓
Four-Corner Detection
    ↓
Corner Ordering
    ↓
Perspective Transformation
    ↓
CLAHE + Sharpening + Gaussian Blur
    ↓
Adaptive Threshold
    ↓
Final Scanned Document
```


## Setup

```bash
git clone https://github.com/<your-username>/document-scanner.git
cd document-scanner
python -m venv venv
```


### Windows

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```


## Usage

### Scan one image

```bash
python main.py --image samples/test1.jpg
```


### Scan all sample images

```bash
python main.py --all
```



## Output

Generated files are saved in `outputs/`:

```text
<name>_contour.jpg
<name>_warped.jpg
<name>_enhanced.jpg
<name>_scanned.jpg
```

- **Contour:** detected document boundary
- **Warped:** perspective-corrected document
- **Enhanced:** sharpened color document
- **Scanned:** final black-and-white scan


## Testing

Run the unit tests with:

```bash
python -m pytest tests/ -v
```


## Documentation

The `docs/` folder contains:

- System Architecture
- Processing Workflow
- Class Diagram
- Sequence Diagram
