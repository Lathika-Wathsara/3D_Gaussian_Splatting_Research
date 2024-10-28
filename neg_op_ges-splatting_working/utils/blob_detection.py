# Difference of Gaussian method
# Code by lathika

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_dog

def detect_blobs_dog(image, min_blob_size, max_blob_size, num_intervals):
    """
    Detects blobs in an image using Difference of Gaussian (DoG).

    Parameters:
        image (ndarray): Input image read using OpenCV (BGR format).
        min_blob_size (float): Minimum size (in pixels) of blobs to detect.
        max_blob_size (float): Maximum size (in pixels) of blobs to detect.
        num_intervals (int): Number of intervals to divide the blob size range.

    Returns:
        blobs (ndarray): Array of detected blobs, with each blob represented as [y, x, r].
                         (y, x) are the centroid coordinates, and r is the radius.
        output_image (ndarray): Image with detected blobs drawn.
    """
    # Convert image to grayscale if it is not already
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Detect blobs using the Difference of Gaussian (DoG) method
    blobs = blob_dog(gray_image, 
                     min_sigma=min_blob_size / np.sqrt(2), 
                     max_sigma=max_blob_size / np.sqrt(2), 
                     sigma_ratio=1.6, 
                     threshold=0.01)
    
    # Compute radii in the 3rd column as sigma * sqrt(2)
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    
    # Draw blobs on the image for output
    output_image = image.copy()
    height, width = gray_image.shape    # To shift the center of origine to bottom left
    for blob in blobs:
        y, x, r = blob
        y = max(height - y,0)   # To shift the center of origine to bottom left
        cv2.circle(output_image, (int(x), int(y)), int(r), (0, 255, 0), 2)

    return blobs, output_image

"""
# Example of how to use the function
# Load image using OpenCV
image_path = "/home/lathika/Workspace/Data_Sets/My_Data_sets/Chair/input/0034.jpg"
image = cv2.imread(image_path)

# Define blob detection parameters
min_blob_size = 5  # Minimum blob size (in pixels)
max_blob_size = 8  # Maximum blob size (in pixels)
num_intervals = 5  # Number of intervals to divide the size range

# Detect blobs 
blobs, output_image = detect_blobs_dog(image, min_blob_size, max_blob_size, num_intervals)


plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.show()

# Output the blobs details: centroid (x, y) and radius
for i, blob in enumerate(blobs):
    print(f"Blob {i+1}: Centroid = ({blob[1]:.2f}, {blob[0]:.2f}), Radius = {blob[2]:.2f}")

    """
