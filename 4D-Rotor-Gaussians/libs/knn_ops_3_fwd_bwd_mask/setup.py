# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

import glob

from setuptools import setup

from torch.utils.cpp_extension import CUDAExtension, BuildExtension

sources = glob.glob("*.cpp") + glob.glob("*.cu")

setup(
    name="knn_ops",
    version="0.1",
    author="wenzheng chen",
    author_email="wenzchen@nvidia.com",
    description="cuda knn accleration operations",
    long_description="cuda knn accleration operations",
    ext_modules=[
        CUDAExtension(
            name="knn_ops",
            sources=sources,
            extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
