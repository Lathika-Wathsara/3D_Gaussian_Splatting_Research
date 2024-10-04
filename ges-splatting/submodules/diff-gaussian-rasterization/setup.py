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

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# Code by lathika (commented the below lines)
"""
os.path.dirname(os.path.abspath(__file__))


setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
"""

# Code by lathika (full code belove was added by me)
# Get the flag from environment variable or defaulr to normal gaussian splatting
folder_flag = os.environ.get('RASTERIZER_FLAG', 'non')

# Construct the folder path based on the flag
if folder_flag == 'non':
    folder_name = "cuda_rasterizer"
elif folder_flag == 'neg_op':
    folder_name = "cuda_rasterizer_for_neg_opacities"
else :
     raise ValueError(f"Invalid flag: {folder_flag}. Expected 'non' or 'neg_op'.")

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                f"{folder_name}/rasterizer_impl.cu",
                f"{folder_name}/forward.cu",
                f"{folder_name}/backward.cu",
                "rasterize_points.cu",
                "ext.cpp"
            ],
            extra_compile_args={
                "nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

