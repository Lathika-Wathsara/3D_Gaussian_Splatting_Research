#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

cmake_minimum_required(VERSION 3.20)

project(DiffRast LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


/*
add_library(CudaRasterizer
	cuda_rasterizer/backward.h
	cuda_rasterizer/backward.cu
	cuda_rasterizer/forward.h
	cuda_rasterizer/forward.cu
	cuda_rasterizer/auxiliary.h
	cuda_rasterizer/rasterizer_impl.cu
	cuda_rasterizer/rasterizer_impl.h
	cuda_rasterizer/rasterizer.h
)

set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "75;86")

# Dynamically set the include directories based on the folder name
target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
target_include_directories(CudaRasterizer PRIVATE third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

*/

# Code by lathika
# Default to use cuda_rasterizer if no flag is set
set(folder_name "cuda_rasterizer" CACHE STRING "Folder containing cuda rasterizer files")

# Code by lathika
# Dynamically add the correct files based on the selected folder
add_library(CudaRasterizer
	${folder_name}/backward.h
	${folder_name}/backward.cu
	${folder_name}/forward.h
	${folder_name}/forward.cu
	${folder_name}/auxiliary.h
	${folder_name}/rasterizer_impl.cu
	${folder_name}/rasterizer_impl.h
	${folder_name}/rasterizer.h
)

set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86")

# Dynamically set the include directories based on the folder name
target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/${folder_name})
target_include_directories(CudaRasterizer PRIVATE third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
