# Code by lathika

import torch
import math 
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
import cv2
import numpy as np

# Cnverting and saving functions
def convert_torch_3D_image_to_greyscale(image):
    """
    This accepts 3D torch tensor with values in the range (0,1) and outputs
    a 2D torch tensor with values in the range (0, 255)
    """
    image_cpu  = image.clone().detach().contiguous().cpu().numpy()
    image_cpu = image_cpu.transpose(1,2,0)
    image_cpu = (image_cpu * 255).astype(np.uint8) if image_cpu.max() <=1.0 else image_cpu.astype(np.uint8)
    image_cpu = cv2.cvtColor(image_cpu , cv2.COLOR_BGR2GRAY)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.from_numpy(image_cpu).to(device)#.float()

def save_torch_image_3D(image, path):
    image_cpu  = image.clone().detach().cpu().numpy()
    image_cpu = image_cpu.transpose(1,2,0)
    image_cpu = (image_cpu * 255).astype(np.uint8) if image_cpu.max() <=1.0 else image_cpu.astype(np.uint8)
    image_cpu = cv2.cvtColor(image_cpu , cv2.COLOR_BGR2RGB)
    #image_cpu = cv2.cvtColor((image_cpu * 255).astype(np.uint8) , cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, image_cpu)

def save_torch_image_2D(image, path):
    image_cpu  = image.clone().detach().cpu().numpy()
    # Ensure it's a 2D grayscale image (single channel)
    if len(image_cpu.shape) == 3:
        image_cpu = image_cpu.squeeze(0)  # Remove the channel dimension if present
    image_cpu = (image_cpu * 255).astype(np.uint8) if image_cpu.max() <=1.0 else image_cpu.astype(np.uint8)
    cv2.imwrite(path, image_cpu)

# Feature extraction and key points selection
"""
# getting mean box kernel
# Previouse kernel sizes  k_size = [21, 13, 5,1] 
def conv_and_get_img(kernel_size, img):
    kernel = torch.ones(3, 1, kernel_size, kernel_size)/(kernel_size**2) # Expanded to 3 channels
    kernel = kernel.cuda()
    img = img.unsqueeze(0)
    blurred_img = torch.nn.functional.conv2d(img, kernel, padding = kernel_size//2, groups = 3)
    blurred_img = blurred_img.squeeze(0)
    return blurred_img
    """

def gaussian_kernel(kernel_size, sigma):
    """
    Generates a 2D Gaussian kernel.
    """
    kernel = torch.Tensor([
        [math.exp(-(x**2 + y**2) / (2 * sigma**2)) for x in range(-(kernel_size // 2), kernel_size // 2 + 1)]
        for y in range(-(kernel_size // 2), kernel_size // 2 + 1)
    ])
    kernel /= kernel.sum()  # Normalize the kernel
    return kernel

def apply_gaussian_blur(img, sigma):
    img = img.float()
    kernel_size = int(6 * sigma + 1) if int(6 * sigma + 1) % 2 == 1 else int(6 * sigma) + 1
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(3, 1, 1, 1)  # Repeat for RGB channels, shape (3, 1, kernel_size, kernel_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blurred_img = F.conv2d(img.unsqueeze(0), kernel.to(device), padding=kernel_size // 2, groups=3)  # (B, C, H, W)
    return blurred_img.squeeze(0), kernel_size

# Function to compute Difference of Gaussians (DoG)
def compute_dog(less_blurred, more_blurred):
    dog = less_blurred - more_blurred
    return dog


    
# Getting corresponding pixels with edge features
def find_regions(image, kernel_size):
    """
    Efficiently finds regions with edge features in a grayscale image.
    
    Args:
        image (torch.Tensor): Input grayscale image (H, W) with values in (0, 255).
        kernel_size (int): Size of the region to analyze.

    Returns:
        list: List of (row, col) tuples where edge features are found.
        torch.Tensor: Output binary mask indicating selected regions.
    """
    
    # Padding to ensure image dimensions are multiples of kernel_size
    r, c = image.shape
    pad_r = (kernel_size - r % kernel_size) % kernel_size
    pad_c = (kernel_size - c % kernel_size) % kernel_size
    padded_img = torch.nn.functional.pad(image, (0, pad_c, 0, pad_r))

    # Extract non-overlapping patches of size (kernel_size, kernel_size)
    patches = padded_img.unfold(0, kernel_size, kernel_size).unfold(1, kernel_size, kernel_size)

    # Condition: Select patches where at least 60% of pixels have intensity ≥ 127
    threshold_mask = patches >= 127  # Boolean mask for pixels ≥ 127
    patch_sums = threshold_mask.sum(dim=(-1, -2))  # Count pixels ≥ 127 in each patch
    mask = (patch_sums >= kernel_size * kernel_size * 0.6).to(torch.uint8)  # Apply 60% threshold

    # Compute coordinates of selected regions
    rows, cols = torch.nonzero(mask, as_tuple=True)
    possible_pixels = torch.stack([(rows * kernel_size + kernel_size // 2), 
                                   (cols * kernel_size + kernel_size // 2)], dim=1)

    # Create binary output image
    out_image = torch.zeros_like(padded_img, dtype=torch.uint8)
    out_image[possible_pixels[:, 0], possible_pixels[:, 1]] = 1

    # Remove padding and return results
    return possible_pixels.tolist(), out_image[:r, :c]

"""
# Checking for edge feature availability around a region box (x,y) 
def is_available(img, kernel_size, x, y):
    sum = 0
    for i in range(x*kernel_size, (x+1)*kernel_size):
        for j in range(y*kernel_size, (y+1)*kernel_size):
            if img[i, j] >= 127:
                sum += 1
    if sum >= kernel_size*kernel_size*0.6:  # 60% Allocated
        return True
    else:
        return False

def process_chunk( padded_img, start_row, end_row, kernel_size, out_image):
    posible_pixels = []
    for i in range(start_row, end_row):
        for j in range(padded_img.shape[1]//kernel_size):
            if is_available(padded_img, kernel_size, i, j):
                out_image[i*kernel_size + kernel_size//2, j*kernel_size + kernel_size//2] = 1
                posible_pixels.append((i*kernel_size + kernel_size//2, j*kernel_size + kernel_size//2))
    return posible_pixels

def find_regions(image, kernel_size, num_threads = 16):

    # Padding
    r, c = image.shape
    padded_img = torch.zeros((r + (kernel_size-r%kernel_size), c + (kernel_size-c%kernel_size)))
    padded_img[:r, :c] = image
    out_image = torch.zeros_like(padded_img)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    padded_img = padded_img.to(device)
    out_image = out_image.to(device)
    # Creating utils
    row_chunk_size = (padded_img.shape[0]//kernel_size)//num_threads
    if row_chunk_size < 1:
        row_chunk_size = 1
        num_threads = padded_img.shape[0]//kernel_size
    results = []
    # Managing pool of threads using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        # submitting tasks to thread pool
        for i in range(num_threads):
            start_row = i*row_chunk_size
            end_row = (i+1)*row_chunk_size if i<num_threads -1 else padded_img.shape[0]//kernel_size
            futures.append(executor.submit(process_chunk, padded_img, start_row, end_row, kernel_size,out_image))
        # Collect results when tasks are completed
        for future in futures:
            results.append(future.result())
        results = list(chain.from_iterable(results))
        
        print(f"length of results = {len(results)}")
        print(f"first element of results = {results[0]}")

    return results, out_image.cpu()[:r,:c] #out_image.cpu().numpy()[:r,:c]
"""


def detect_H_freq_with_DoG(image_high, image_low, kernel_size):
    """
    image should be a tensor
    image should have one channel (Grey scale) - (values in between (0,255))
    results: 2d tensor (n,2) with mean values
    kernel_size_vis: for visualization using "apply_mag_kernel" function
    """
    #print(f"image_high = {image_high.shape}")
    device = image_high.device
    image_high = convert_torch_3D_image_to_greyscale(image_high)
    image_low = convert_torch_3D_image_to_greyscale(image_low)
    #print(f"image_high_gray = {image_high.shape}")

    #blurred_img, kernel_size = apply_gaussian_blur(image, sigma)
    #print(f"kernel_size = {kernel_size}")
    dog_image = compute_dog(image_high, image_low)
    
    kernel_size_vis = kernel_size//2 if (kernel_size//2)%2 == 1 else kernel_size//2 + 1   # Kernel to visualize (when makng the gaussinas (for apply_mag_kernel function), we can use this, but we might need to use the lowest sigma value)
    results , out_image = find_regions(dog_image, kernel_size_vis)    #find_regions(dog_image, kernel_size_vis, 32)  here results is a list
    return dog_image, torch.tensor(results, device = device), out_image, kernel_size_vis


# For visualizing
import matplotlib.pyplot as plt


def apply_mag_kernel(image, kernel_size):
    # Get image dimensions
    height, width = image.shape
    
    # Create a padded image to handle the borders
    padded_img = torch.zeros((height + kernel_size - 1, width + kernel_size - 1), dtype=torch.uint8)
    padded_img[:height, :width] = image
    
    # Output image (same size as padded image)
    out_image = torch.zeros_like(padded_img)
    
    # Apply the kernel over the image with stride equal to kernel_size
    for i in range(0, padded_img.shape[0] - kernel_size + 1, kernel_size):
        for j in range(0, padded_img.shape[1] - kernel_size + 1, kernel_size):
            # Extract the current region of the image that corresponds to the kernel
            region = padded_img[i:i+kernel_size, j:j+kernel_size]
            
            # If any element in the region is 1, set the region below the kernel to 1
            if torch.any(region == 1):
                # Set the entire region below the kernel to 1
                out_image[i:i+kernel_size, j:j+kernel_size] = 1
                
    return out_image[0:height, 0:width]  # Return the region of interest, not the padded area

def visualize_images(images, titles):
    fig, axes = plt.subplots(len(images)//3+1, 3, figsize=(15,12)) # plt.subplots(1, len(images), figsize=(20,20))
    for i in range(len(images)):
        axes[i//3, i%3].imshow(images[i], cmap='gray')
        axes[i//3, i%3].set_title(titles[i])
        axes[i//3, i%3].axis('off')
    """
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')"""
    plt.show()


"""
img_path = "/content/drive/MyDrive/Colab Notebooks/Test/Blur/0013.jpg"  # Replace with your image path
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale
sigmas = [1.0, 2**0.5] 

images, dog_image, results, out_image, kernel_size = detect_H_freq_with_DoG(image, sigmas)

print(f"kernel for magnify = {kernel_size}")
magnified = apply_mag_kernel(torch.tensor(out_image),kernel_size)

images.extend([dog_image, out_image, magnified])
titles = ['blur 1', 'blure 2', 'dog', 'out_image', 'magnified']
visualize_images(images, titles)

"""


# Getting 3d points from extracted high frequency points, depths and gaussian indexes

def pix2ndc(pix, s):
    return (2*pix + 1)/s - 1

def get_3d_ind(depth_tensor, mean_pixels_tensor, H, W, ful_proj_mat):
    """
    depth tensor: 1D tensor
    mean_pixel_tensor: 2D tensor (row represent a mean pixel point)
    H, W: Hight and width
    ful_proj_mat: Full projection matrix (column major matrix)
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_tensor = depth_tensor.to(device)
    mean_pixels_tensor = mean_pixels_tensor.to(device)
    ful_proj_mat = ful_proj_mat.to(device)
    ndc_x = pix2ndc(mean_pixels_tensor[:,1], W) # x should be columns in rendering
    ndc_y = pix2ndc(mean_pixels_tensor[:,0], H) 
    x = ndc_x*(depth_tensor + 0.0000001)
    y = ndc_y*(depth_tensor + 0.0000001)
    z = depth_tensor
    cam_coordiantes = torch.stack([x, y, z], dim=1)
    A = ful_proj_mat[:3,[0,1,3]] #ful_proj_mat[:3,:3]
    b = ful_proj_mat[3,[0,1,3]] #ful_proj_mat[3,:3]
    l_2 = len(z)
    A_batch = torch.tile(torch.linalg.inv(A).unsqueeze(0), (l_2,1,1))
    diff_batch = (cam_coordiantes - b).unsqueeze(0).reshape(-1,1,3)
    #print(f"A_batch_shape = {A_batch.shape}, diff_batch_shape = {diff_batch.shape}")
    orig_coordinates = torch.bmm(diff_batch, A_batch).reshape(-1,3)
    return orig_coordinates


# Extracting depth and means

def get_offset(kernel_size = 3):
    k = kernel_size   # Kernel size
    x_offsets, y_offsets = torch.meshgrid(torch.arange(-(k-1)//2, (k-1)//2+1), torch.arange(-(k-1)//2, (k-1)//2+1), indexing="ij")
    offsets = torch.stack((x_offsets.reshape(-1), y_offsets.reshape(-1)), dim=1)
    return offsets

def min_depth_and_idx(depth_arr, means_arr, gaussian_index_tensor):

    """
    depth_arr and means_arr are 2d tensors
    This will return a depth corresponding to means and their surrounding window and their indexes
    mask: (H,W)boolian tensor with true on selected residual points (corresponding to min values)
    """
    means_arr = means_arr.long()    # To make sure its integer
    H, W = depth_arr.shape
    num_points = means_arr.shape[0]
    offsets = get_offset().to(means_arr.device)
    # To make window coordinates of size (num_points, kernel_size^2, 2)
    window_coordinates = means_arr.unsqueeze(1) + offsets.unsqueeze(0) 
    y_coords = window_coordinates[:,:,0].clamp(0, H-1)  # Clamping to make sure indeces are inside the boundary
    X_coords = window_coordinates[:,:,1].clamp(0, W-1)
    window_vals = depth_arr[y_coords, X_coords] # Shape (num_points, kernel_size^2)
    min_vals, min_idx = torch.min(window_vals, dim=1)   # Getting mins and indices
    # Getting min values indices
    min_offsets = offsets[min_idx]
    min_coords = means_arr + min_offsets
    # Again clamping to make sure no out of bounds
    y_min = min_coords[:,0].clamp(0, H-1)
    x_min = min_coords[:,1].clamp(0, W-1)
    # Mask - test can remove
    mask = torch.zeros_like(depth_arr, dtype=torch.bool)
    mask[y_min, x_min] = True
    # Index_values
    gauss_idx_vals =  gaussian_index_tensor[y_min,x_min]
    return min_vals, gauss_idx_vals ,mask 

# Final function for 3d original points and corresponding gaussian index array

def get_3d_points_and_gaussian_index(depth_tensor, gaussian_index_tensor, mean_pixels_tensor, full_proj_mat, portion = 0.2):
    """
    inputs:-
    depth_tensor: 2D (H,W) image like depth tensor
    gaussian_index_tensor: 2D image like gaussian index tensor 
                            (each index is the index of the prominent gaussian of thet pixel)
    mean_pixels_tensor: 2D tensor (row represent a mean pixel point)
    full_proj_mat: Full projection matrix (column major matrix)
    portion: The portion we use from the whole set (value between 0,1)

    Outputs:-
    orig_coordinates: 3D mean points in original coordinate system
    gaussian_indexes: (1D tensor) selected gaussian indexes (corresponds to the residual requirments)
    """

    # Selecting a portion
    l = len(mean_pixels_tensor)
    p_mask = torch.randperm(l)[:int(l*portion)]
    mean_pixels_tensor = mean_pixels_tensor[p_mask]

    # Get depth tensor (depths of residual points) and masks for gauss index
    depth_values, gaussian_indexes ,mask = min_depth_and_idx(depth_tensor, mean_pixels_tensor, gaussian_index_tensor)
    H, W = depth_tensor.shape
    #gaussian_indexes = gaussian_index_tensor[mask]  # Getting prominent gaussian indexes (indexes in gaussian model)

    # Get original 3d coordinates
    orig_coordinates = get_3d_ind(depth_values, mean_pixels_tensor, H, W, full_proj_mat)

    return orig_coordinates, gaussian_indexes.long()


# Get scales in world coordinated

def get_world_scales(sigma, view_mat, tan_fovx, tan_fovy, means_3d, H, W):
    """
    Inputs
        sigma: std of high freq gauss
        view_mat: view matrix
        tan_fov: field of view tans
        means_3d: gaussian means in world coordinates (2d matrix (n,3))
    Output
        world_scales: scales in world coordinates (2d matrix with size (n,3))
    """

    device = means_3d.device
    n = len(means_3d)
    p_hom = torch.cat((means_3d, torch.ones((n, 1), device = device)), dim=1)
    cam_coord = (p_hom @ view_mat)[:,:3] #torch.bmm(p_hom.unsqueeze(1), torch.tile(view_mat.unsqueeze(0),(n,1,1))).squeeze()[:,:3]
    depth = cam_coord[:,2]    # To get a 2D tensor (cam_coord[:,2] will give a 1D tensor)
    focal_x, focal_y =  W/(2*tan_fovx), H/(2*tan_fovy)
    lim_x = 1.3* tan_fovx
    lim_y = 1.3* tan_fovy
    c_x = torch.clamp(cam_coord[:,0]/depth, -lim_x, lim_x) * depth
    c_y = torch.clamp(cam_coord[:,1]/depth, -lim_y, lim_y) * depth
    # Getting Jacobian
    J = torch.zeros((n, 2, 3), device=device)   # (n, 3, 3) # Row major
    J[:, 0, 0] = focal_x / depth    #.squeeze()
    J[:, 1, 1] = focal_y / depth
    #print((focal_x * c_x).squeeze().shape)
    #print((depth**2).shape)
    J[:, 0, 2] = -(focal_x * c_x).squeeze() / (depth**2)
    J[:, 1, 2] = -(focal_y * c_y).squeeze() / (depth**2)
    
    W = view_mat[:3,:3].T   # To make row major
    T = torch.bmm(J, W.expand(n,3,3))   # Transformation matrix

    # Get cov_2D screen space
    cov_2D = torch.zeros((n,2,2), device=device) # (n,3,3)
    cov_2D[:,0,0] = sigma**2
    cov_2D[:,1,1] = sigma**2

    # Get 3D cov
    p_inv_T =  torch.linalg.pinv(T)
    s_sq = torch.bmm(torch.bmm(p_inv_T, cov_2D), p_inv_T.transpose(1,2))
    s_sq = torch.diagonal(torch.clamp(s_sq, min=1e-6), dim1=1, dim2=2)
    s = torch.sqrt(s_sq)
    return s