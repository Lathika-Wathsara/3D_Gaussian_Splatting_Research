/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */




#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// Code by lathika - test
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>
#include <cuda_runtime.h>
void checkMemoryLocation(void* ptr) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
    if (err != cudaSuccess) {
        std::cout << "Error retrieving pointer attributes: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    #if CUDART_VERSION >= 10000  // Check if CUDA version is 10.0 or later
        if (attributes.type == cudaMemoryTypeDevice) {
            std::cout << "Pointer is in GPU (device) memory." << std::endl;
        } else if (attributes.type == cudaMemoryTypeHost) {
            std::cout << "Pointer is in CPU (host) memory." << std::endl;
        } else if (attributes.type == cudaMemoryTypeManaged) {
            std::cout << "Pointer is in unified (managed) memory." << std::endl;
        } else {
            std::cout << "Unknown memory location." << std::endl;
        }
    #else
        if (attributes.memoryType == cudaMemoryTypeDevice) {
            std::cout << "Pointer is in GPU (device) memory." << std::endl;
        } else if (attributes.memoryType == cudaMemoryTypeHost) {
            std::cout << "Pointer is in CPU (host) memory." << std::endl;
        } else {
            std::cout << "Unknown memory location." << std::endl;
        }
    #endif
}


// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Code by lathika
// We gat cov 3D matrix with respect to camera position
// Then get the variance from the look at dirtection  
__device__ float get_lookat_var(const float* cov3D, 
								const float* viewmatrix)
{
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);
	
	glm::mat3 cov = glm::transpose(W) * glm::transpose(Vrk) * W;

	return float(cov[0][0]);
}


// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	bool* ctn_gauss_mask,	// Code by lathika
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float* look_at_var_arr,	// Code by lathika
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;
	ctn_gauss_mask[idx] = false; // Code by lathika (This will be updated in render, for backward pass)

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Code by lathika
	// Get the variance of 3D gaussian wrt the look at direction
	float look_at_var = get_lookat_var(cov3D, viewmatrix);
	look_at_var_arr[idx] = look_at_var;

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	bool* ctn_gauss_mask,	// Code by lathika
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	int* __restrict__ unwanted_gauss,	// Code by lathika
	const float* __restrict__ depths,	// Code by lathika
	const float* __restrict__ look_at_var_arr,	// Code by lathika
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float*  __restrict__  alpha_sum,	// Code by lathika
	uint32_t*  __restrict__ contrib_count)	// Code by lathika
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Code by lathika
	const int Max_neg_arr_size = BLOCK_SIZE*8 ;//Changed const int to int and   BLOCK_SIZE*3 (not enough) into (( range.y - range.x + 32 -1)/32)*32 into BLOCK_SIZE*5 changed by lathika
	//uint32_t neg_gaus_arr[Max_neg_arr_size];
	int neg_gaus_idx_arr[Max_neg_arr_size];
	float neg_gaus_alpha_arr[Max_neg_arr_size];
	float neg_gaus_depths[Max_neg_arr_size];
	float neg_gaus_look_at_var[Max_neg_arr_size];
	uint32_t num_neg_gauss = 0;
	int pointer = -1;
	//__shared__ uint32_t neg_gaus_arr_in_block[BLOCK_SIZE];
	//__shared__ float neg_gaus_alpha_arr_in_block[BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];
	__shared__ float collected_look_at_var[BLOCK_SIZE];

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{	
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];

			// Code by lathika
			collected_depths[block.thread_rank()]  = depths[coll_id];
			collected_look_at_var[block.thread_rank()] = look_at_var_arr[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{	
			// Keep track of current position in range
			contributor++;

			// Code by lathika
			

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;


			// Code by lathika - test
			/*
			// Allocate memory for random states on device
			// Retrieve the random state for this thread
			curandState state;
    		curand_init(1234, pix_id, 0, &state);
			// Generate a random integer, e.g., between 1 and 100
    		int randomInt = curand(&state) % 100000000;
			if (randomInt%99999999==0){
				printf("\n opacity = %f",con_o.w);
				printf("\n rand num = %d", randomInt);
			} */

			// Code by lathika
			bool is_neg_gauss_used = false;
			float alpha;	// Since alpha is use in two places, I initially declared it here
			// If a gaussian is negetive, add it to the array, and continue (No need to run the parts bellow)
			if (con_o.w < 0.0f)
			{	
				alpha = max(-0.99f, con_o.w * exp(power));
				if (alpha > -1.0f / 255.0f)
					continue;
				if (pointer < Max_neg_arr_size - 1) {  // Check for array bounds before incrementing
					pointer++;
					num_neg_gauss++;
					neg_gaus_idx_arr[pointer] = collected_id[j];
					neg_gaus_alpha_arr[pointer] = alpha;
					neg_gaus_depths[pointer] = collected_depths[j];
					neg_gaus_look_at_var[pointer] = collected_look_at_var[j];
				} else {
					// Handle overflow case: you could break, skip, or flag this situation
					// You could set a flag to track that you've run out of space
					//printf("Pointer val = %d \n",pointer);
					//printf("num_neg_gauss = %d \n",num_neg_gauss);
					//printf("Total gauss for a block = %d\n", range.y - range.x);
					//printf("Warning: Maximum number of negative Gaussians reached. Skipping additional Gaussians.\n");
					pointer = -1;	// Code by lathika test 
					num_neg_gauss = 0;	// Code by lathika test
				}
				continue;
				}
			// Code by lathika
			alpha = min(0.99f, con_o.w * exp(power));
			if (num_neg_gauss !=0)
			{	
				
				float pos_var = collected_look_at_var[j];
				float pos_depth = collected_depths[j];
				float T_neg = 1.0f;
				for (int u= pointer; u > -1; u--)
				{
					float neg_var = neg_gaus_look_at_var[u];
					float neg_depth = neg_gaus_depths[u];
					float neg_alpha = neg_gaus_alpha_arr[u];
					//printf("pos_depth = %.4f\n",pos_depth);
					//printf("neg_depth = %.4f\n",neg_depth);
					//printf("glm::sqrt(pos_var) = %.4f \n",glm::sqrt(pos_var));
					//printf("glm::sqrt(neg_var) = %.4f\n",glm::sqrt(neg_var));
					if (abs(pos_depth-neg_depth) > std_diff_coff*(glm::sqrt(pos_var)+glm::sqrt(neg_var)))
					{	
						//printf("\n Enter pointer reduce\n");
						if (pointer == u)
						{
							pointer = -1;
							num_neg_gauss=0;
						}
						else
						{
							// #include <thrust/copy.h> // Code by lathika (Add before the function)
							//thrust:: copy(neg_gaus_look_at_var+u, neg_gaus_look_at_var + pointer +1, neg_gaus_look_at_var);
							//thrust:: copy(neg_gaus_depths + u +1, neg_gaus_depths + pointer +1, neg_gaus_depths);
							for (int v = u+1; v < pointer+1; v++){
								//int idx = neg_gaus_idx_arr[v];
								//unwanted_gauss[idx] = 1;	// Commented by lathika , this is done below after confirming thatthe pos gauss is used
								neg_gaus_idx_arr[v-u-1] = neg_gaus_idx_arr[v];
								neg_gaus_look_at_var[v-u-1] = neg_gaus_look_at_var[v];
								neg_gaus_depths[v-u-1] = neg_gaus_depths[v];
								neg_gaus_alpha_arr[v-u-1] = neg_gaus_alpha_arr[v];
							}
							pointer = pointer -u -1;
							num_neg_gauss = pointer + 1;
							is_neg_gauss_used = true;
						}
						break;
					}
					else
					{
						alpha = alpha + neg_alpha*T_neg;
						T_neg = T_neg*(1 + neg_alpha);
					}
				}
				ctn_gauss_mask[collected_id[j]] = alpha < 1.0f / 255.0f && alpha + 1.0f - T_neg >  1.0f / 255.0f;	// Code by lathika
				
			}	

			// Code by lathika
			atomicAdd(&alpha_sum[collected_id[j]], alpha);  // Accumulate alpha
			atomicAdd(&contrib_count[collected_id[j]], 1);  // Count contributions


			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// float alpha = min(0.99f, con_o.w * exp(power));	// Code by lathika
			if (alpha < 1.0f / 255.0f)	
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			if (T < 1e-8) T = 0.0f;		// Code by lathika (to mitigate underflow)

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;

			// Code by lathika - If the neg gaussed were used - if the positive gaussians that were effected by the neg gauss were used (not continued the loop)
								// Then the corresponding negative gauss should considered as wanted gauss
			unwanted_gauss[collected_id[j]]=1;	// If a positive gaussian has a valid positive opacity, then add it to the filter
			if (is_neg_gauss_used){
				for (int u= pointer; u > -1; u--){
					int idx = neg_gaus_idx_arr[u];
					unwanted_gauss[idx]=1;	// 1 means, this gaussian should be considered
				}
			}
			

		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{

		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}


void FORWARD::render( int P,	// Code by lathika (int P, to get the numper of gaussians)
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	bool * ctn_gauss_mask,	// Code by lathika
	const float2* means2D,
	const float* colors,
	int* unwanted_gauss,	// Code by lathika
	const float* depths,	// Code by lathika
	const float* look_at_var_arr, // Code by lathika
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{

	// Code by lathika - Globaly initializing arrays to compute the alpha values to sort out the unwanted gaussians
	// Allocated memory for alpha_sum and contrib_count and unwanted_gauss
	float* alpha_sum;
	uint32_t* contrib_count;
	int n_gaussians = P;
	int* unwanted_gauss_both;
	// Allocate alpha_sum & contrib_count as arrays on the device
	gpuErrorchk(cudaMalloc(&alpha_sum, n_gaussians * sizeof(float)));	// alpha_sum - will store the sum of alpha values to calculate the average and add gaussians to prune.
	gpuErrorchk(cudaMalloc(&contrib_count, n_gaussians * sizeof(uint32_t))); // contrib_count - will store the number of contribution for each gaussian
	// Initializing 
	gpuErrorchk(cudaMemset(alpha_sum, 0, n_gaussians*sizeof(float)));
	gpuErrorchk(cudaMemset(contrib_count, 0, n_gaussians*sizeof(uint32_t)));

	gpuErrorchk(cudaMallocManaged(&unwanted_gauss_both, n_gaussians * sizeof(int)));	// Allocate Unified Memory for GPU/Host updates
	gpuErrorchk(cudaMemset(unwanted_gauss_both, 0, n_gaussians * sizeof(int)));
	/*// Initialize unwanted_gauss_both on the host - Remove this part if wants
    for (int i = 0; i < n_gaussians; i++) {
        unwanted_gauss_both[i] = 0;
    }*/

	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		ctn_gauss_mask,	// Code by lathika
		means2D,
		colors,
		unwanted_gauss_both,	// Code by lathika
		depths,		// Code by lathika
		look_at_var_arr, // Code by lathika
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		alpha_sum,	// Code by lathika
		contrib_count);	// Code by lathika

	// Code by lathika
	// Check for kernel launch errors
    gpuErrorchk(cudaPeekAtLastError());
    gpuErrorchk(cudaDeviceSynchronize()); // Ensure kernel execution is complete - Wait for GPU to finish
	// Host arrays to store alpha_sum and contrib_count

	float* h_alpha_sum = new float[n_gaussians];
	uint32_t* h_contrib_count = new uint32_t[n_gaussians];

    // Copy results back from device to host
    gpuErrorchk(cudaMemcpy(h_alpha_sum, alpha_sum, n_gaussians * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrorchk(cudaMemcpy(h_contrib_count, contrib_count, n_gaussians * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    gpuErrorchk(cudaMemcpy(unwanted_gauss, unwanted_gauss_both, n_gaussians * sizeof(int), cudaMemcpyDeviceToHost));

	// Calculating the average and updating the filter
	float alpha_threshold = 1.0f / 255.0f;// Change this accordingly
	for (int i = 0; i < n_gaussians; i++) {
		if (h_contrib_count[i] > 0 ) {  // Make sure there's at least one contribution
			float avg_alpha = h_alpha_sum[i] / h_contrib_count[i];
			if (avg_alpha > alpha_threshold) {  // Apply your threshold
				unwanted_gauss[i] = 1;  // Mark Gaussian as wanted. If a gaussian has a valid positive opacity, then add it to the filter
			}
		}
	}

	// cudaMemcpy(d_unwanted_neg_gauss, unwanted_neg_gauss, n_gaussians * sizeof(int), cudaMemcpyHostToDevice);

	// Code by lathika - Free memory after kernel execution
    gpuErrorchk(cudaFree(alpha_sum));
    gpuErrorchk(cudaFree(contrib_count));
    gpuErrorchk(cudaFree(unwanted_gauss_both));
	// Free host memory
	delete[] h_alpha_sum;
	delete[] h_contrib_count;
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	bool* ctn_gauss_mask,	// Code by lathika
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float* look_at_var_arr, // Code by lathika
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		ctn_gauss_mask,	// Code by lathika
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		look_at_var_arr, // Code by lathika
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}