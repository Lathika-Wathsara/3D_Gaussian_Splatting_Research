# Difference of Gaussian method
# Code by lathika

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_dog
import torch
import math

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
    blobs_xy_new = torch.ones_like(blobs_xy).to(device='cpu')
    blobs_xy_new[:,0], blobs_xy_new[:,1] = pix2ndc(blobs_xy[:,0],w), pix2ndc(blobs_xy[:,1],h)

    p = projection_full.T[torch.tensor([0,1,3])][:,:3].cpu().to(dtype=torch.float)  # In 3d gaussian splatting, they have used column prioritizing
    b = np.ones((size,3)) # This is b matrix in Ax = b
    b[:,:2] = blobs_xy_new.detach()*depth.reshape(size,1).cpu().detach().numpy()
    b[:,2] =  depth.cpu().detach().numpy()
    b = b - projection_full.T[torch.tensor([0,1,3])][:,3].cpu().detach().numpy()
    b = torch.tensor(b, dtype=torch.float)
    X = np.ones((size,3))
    for i in range(size):
        X[i], res, r, s = torch.linalg.lstsq(p, b[i], rcond=None) # Solving Ax = b for x
    return torch.tensor(X) 


# Optimized for cuda
def get_mean_3d_cuda(blobs_xy, depth, projection_full, w, h):
    """
    Inputs:
        blobs_xy - 2D pixel centroid values of blobs
        depth - Average depth tensor (use knn to get neighbors and get an average depth value for blobs)
        projection_full - Projection matrix of camera viewpoint (projection_mat @ view_mat)
        w - Image width
        h - Image height
    Output:
        X - 3D coordinates with respect to world coordinates
    """
    device = blobs_xy.device  # Assuming blobs_xy, depth, and projection_full are on the same device (CUDA)

    size = blobs_xy.shape[0]
    # Converting pixel values to ndc
    blobs_xy_new = torch.ones_like(blobs_xy, device=device)
    blobs_xy_new[:, 0], blobs_xy_new[:, 1] = pix2ndc(blobs_xy[:, 0], w), pix2ndc(blobs_xy[:, 1], h)

    # Preparing matrices for Ax = b
    p = projection_full.T[torch.tensor([0, 1, 3], device=device)][:, :3]  # In 3d gaussian splatting, they have used column prioritizing
    b = torch.ones(size, 3, device=device)   # This is b matrix in Ax = b
    b[:, :2] = blobs_xy_new * depth.view(size, 1)
    b[:, 2] = depth
    b = b - projection_full.T[torch.tensor([0, 1, 3], device=device)][:, 3]

    # Solve Ax = b in batch mode
    # p needs to be expanded to match the size of b for batch lstsq
    p_batch = p.unsqueeze(0).expand(size, -1, -1)  # Expanding p to shape (size, 3, 3)
    b_batch = b.unsqueeze(-1)  # Making b shape (size, 3, 1)

    # Use torch.linalg.lstsq on CUDA with the `gels` driver (default for CUDA)
    X= torch.linalg.lstsq(p_batch, b_batch).solution     # Solving Ax = b for x

    # Remove the extra dimension added for lstsq compatibility
    X = X.squeeze(-1)

    return X


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

# Calculating the scaling factor and converting pixel radius into scalings
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def get_scalings(viewpoint_cam, rad_pix, width, height, depth):
    focal_len_x_pixels = fov2focal(viewpoint_cam.FoVx, width)
    focal_len_y_pixels = fov2focal(viewpoint_cam.FoVy, height)
    focal_len_pixels = 0.5*(focal_len_x_pixels + focal_len_y_pixels)  
    scale_factor = depth / focal_len_pixels     # For pinhole camera
    blob_scale = scale_factor*rad_pix
    return blob_scale


# Code by lathika - For initializing the negative gaussians
# Detect and get neg gauss points
def get_new_neg_points(gt_image, viewpoint_cam, means_2D, depths):
    image_cpu_numpy = gt_image.clone().detach().cpu().numpy()   # Shape = (3,H,W)
    image_cpu_numpy = np.transpose(image_cpu_numpy, axes=(1,2,0))
    # Define blob detection parameters
    min_blob_size = 5  # Minimum blob size (in pixels)
    max_blob_size = 8  # Maximum blob size (in pixels)
    num_intervals = 5  # Number of intervals to divide the size range
    # Detect blobs 
    blobs, _ = detect_blobs_dog(image_cpu_numpy, min_blob_size, max_blob_size, num_intervals)
    blobs = torch.tensor(blobs, device="cuda")
    blobs_xy = blobs[:,:2]

    # Filtering the blobs closer to the edges
    height = viewpoint_cam.image_height
    width  = viewpoint_cam.image_width
    blobs_mask = get_border_blob_mask(blobs_xy, width, height)
    blobs_filtered = blobs[blobs_mask]
    blobs_xy = blobs_filtered[:,:2]
    blob_radii = blobs_filtered[:,2]
    blob_rotation = torch.zeros(blobs_xy.shape[0],4)
    blob_rotation[:,0] = 1

    # Filter out (0,0) means_2d values (because gaussians are out of frustrum)
    means_2D_mask = get_in_image_gaussian_mask(means_2D)
    means_2D_new = means_2D[means_2D_mask]
    depths_new = depths[means_2D_mask]
    
    # Calculating knn
    knn_indices , _ = knn_2d(blobs_xy, means_2D_new, k=5)
    depths_avg_new = torch.mean(depths_new[knn_indices],dim=1)

    # Getting scales
    blob_scales = get_scalings(viewpoint_cam, blob_radii, width, height, depths_avg_new)
    blob_scales_tensor = torch.reshape(blob_scales,(blob_scales.shape[0],1)).repeat(1,3)

    # Getting 3d coordinates
    neg_gaus_3d_means =  get_mean_3d_cuda(blobs_xy, depths_avg_new, viewpoint_cam.full_proj_transform, width, height)

    return neg_gaus_3d_means, blob_scales_tensor, blob_rotation


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
