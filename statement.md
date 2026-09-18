# Problem Statement

People often use their phones to take pictures of documents instead of using a scanner. These pictures might be tilted, have background and have a warped look, which makes the document not very useful for direct use.

This project creates a command-line document scanner that finds the document area in a picture finds its four corners, fixes the perspective and improves the result to get a more like a scanned document.

## Scope

- Finds one document and its four corners in a picture

- Makes the document straight using perspective transformation

- Makes the transformed document by increasing contrast and sharpening

- Makes a black-and-white version of the document using adaptive thresholding

- Works with one picture and with multiple pictures through the command line

- Not included: documents with more than one page, paper that is curved or very bent and making a mobile app

## Target Users

- Students and people who need a simple command-line document scanning tool

- Students who are learning traditional computer vision methods like segmenting, finding contours finding corners and using perspective transformation

## High-Level Features

1. **Detection module**. Changes the image does Canny edge detection uses GrabCut-based segmenting to get the document area and finds the four corners of the document.

2. **Transform module**. Arranges the four corners found and uses perspective transformation to make the document straight and flat.

3. **Enhancement module**. Changes the transformed document to black and white improves contrast, in parts using CLAHE sharpens the image uses Gaussian blur to remove noise and uses adaptive thresholding to get a black and white scanned version.
