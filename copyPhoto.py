from google.colab import files
import cv2
import numpy as np

# Load the image
image = cv2.imread("./traffic_sign.png")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to binarize
_, binarized = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for index, contour in enumerate(contours):
    # Get bounding box from the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Extract the region of interest
    sign = image[y:y+h, x:x+w]

    # Convert to RGB (OpenCV uses BGR by default)
    sign_rgb = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)

    # Create a binary mask where white regions are set to 0 and non-white regions are set to 1
    lower_white = np.array([220, 220, 220], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(sign_rgb, lower_white, upper_white)
    mask_inv = cv2.bitwise_not(mask)

    # Convert image to RGBA format and set the alpha channel to the inverse of the mask
    sign_rgba = cv2.cvtColor(sign_rgb, cv2.COLOR_RGB2BGRA)
    sign_rgba[:, :, 3] = mask_inv

    # Save the sign with a transparent background
    cv2.imwrite(f"sign_{index}.png", sign_rgba)
    files.download(f'sign_{index}.png')
