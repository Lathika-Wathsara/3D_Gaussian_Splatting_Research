# Difference of Gaussian method
# Code by lathika

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_dog
import torch

def knn_2d(blob_xy, gauss_xy, k):
    device = gauss_xy.device
    blob_xy = blob_xy.to(device)
    gauss_xy = gauss_xy.to(device)
    dists = torch.cdist(blob_xy.float(), gauss_xy.float(), p=2)  # Compute pairwise Euclidean distances
    knn_dists, knn_indices = torch.topk(dists, k, dim=1, largest=False)
    return knn_indices, knn_dists


def pix2ndc(p,s):
    return (2*p +1)/s -1


def get_mean_3d(blobs_xy, depth, projection_full, w, h):
    """
    Inputs:-
        mean_2d - 2d prixel centroid values of blobs
        depth - Average depth tensor (use knn to get the neighbours and get an average depth value for blobs)
        projection_full - Projection_full matrix of camera viewpoint (projection_mat @ view_mat)
        w - image width
        h - image hight
    Output
        X - 3d coordinates w.r.t world coordinates
    """
    size = blobs_xy.shape[0]
    # Converting pixel values to ndc
    blobs_xy_new = torch.ones_like(blobs_xy)
    blobs_xy_new[:,0], blobs_xy_new[:,1] = pix2ndc(blobs_xy[:,0],w), pix2ndc(blobs_xy[:,1],h)

    p = projection_full.T[torch.tensor([0,1,3])][:,:3]  # In 3d gaussian splatting, they have used column prioritizing
    b = torch.ones(size,3)  # This is b matrix in Ax = b
    b[:,:2] = blobs_xy_new.detach()*depth.reshape(size,1)
    b[:,2] =  depth
    b = b - projection_full.T[torch.tensor([0,1,3])][:,3]
    X = np.ones((size,3))
    for i in range(size):
        X[i], res, r, s = torch.linalg.lstsq(p, b[i], rcond=None) # Solving Ax = b for x
    return torch.tensor(X, requires_grad=True) 


def get_border_blob_mask(blobs_xy, w, h):

    device = blobs_xy.device

    # We take the blobs in the range of (w*0.01, h*0.01) < (x,y) < (w*0.99, h*0.99)
    w_min, w_max = int(w*0.01), int(w*0.99) 
    h_min, h_max = int(h*0.01), int(h*0.99)
    mask = torch.logical_and(torch.tensor([w_max,h_max],device=device)>blobs_xy ,blobs_xy>torch.tensor([w_min,h_min],device=device))
    mask = torch.logical_and(mask[:,0], mask[:,1])

    return mask


def get_in_image_gaussian_mask(means_2d):
    # We take only the positive points to calculated knn (to make things faster)
    mask = torch.logical_and(0.0 < means_2d[:,0] ,0.0 < means_2d[:,1])
    return mask

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
    # x, y correction (the output of blob_dog is like (y,x,r))
    blobs_new = np.empty_like(blobs)
    blobs_new[:,0], blobs_new[:,1], blobs_new[:,2] = blobs[:,1], height-blobs[:,0], blobs[:,2]

    for blob in blobs:
        y, x, r = blob
        y = max(height - y,0)   # To shift the center of origine to bottom left
        cv2.circle(output_image, (int(x), int(y)), int(r), (0, 255, 0), 2)

    return blobs_new, output_image

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
