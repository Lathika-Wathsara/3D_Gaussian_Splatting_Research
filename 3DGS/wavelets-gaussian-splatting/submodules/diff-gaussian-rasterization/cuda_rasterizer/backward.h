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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float3* conic_freq,		// Code by lathika
		const glm::vec2* freq_coeff,	// Code by lathika
		const float* colors,
		const float* depths,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_invdepths,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float4* dL_dconic_freq,			// Code by lathika
		float* dL_dopacity,
		glm::vec2* dL_dfreq_coeff,		// Code by lathika
		float* dL_dcolors,
		float* dL_dinvdepths,
		bool active_wavelets);			// Code by lathika

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const float* opacities,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const glm::vec3* scales_freq,		// Code by lathika
		const glm::vec4* rotations_freq,	// Code by lathika
		const float scale_modifier,
		const float* cov3Ds,
		const float* cov3D_freq_ptr,		// Code by lathika
		const float* view,
		const float* proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		const float3* dL_dmean2D,
		const float* dL_dconics,
		const float* dL_dconic_freq,		// Code by lathika
		const float* dL_dinvdepth,
		float* dL_dopacity,
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_dcov3D,
		float* dL_dcov3D_freq,				// Code by lathika
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot,
		glm::vec3* dL_dscale_freq,			// Code by lathika
		glm::vec4* dL_drot_freq,			// Code by lathika
		bool antialiasing,
		bool active_wavelets);				// Code by lathika
}

#endif
