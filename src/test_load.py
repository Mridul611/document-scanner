import cv2

image = cv2.imread("samples/test1.jpg")

if image is None:
    print("Failed to load image. Check the file path.")
else:
    print("Image loaded successfully!")
    print("Image dimensions (height, width, channels):", image.shape)
