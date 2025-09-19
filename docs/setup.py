from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import os

# 获取依赖路径
pybind11_include = pybind11.get_include()
try:
    import torch
    torch_include = torch.utils.cpp_extension.include_paths()
except:
    torch_include = [
        "/opt/conda/envs/base_222/lib/python3.10/site-packages/torch/include",
        "/opt/conda/envs/base_222/lib/python3.10/site-packages/torch/include/torch/csrc/api/include"
    ]
cuda_include = os.environ.get("CUDA_INCLUDE_DIR", "/usr/local/cuda/include")
cuda_lib = os.environ.get("CUDA_LIB_DIR", "/usr/local/cuda/lib64")

# 扩展模块（关键：extra_compile_args使用列表而非字典）
ext_modules = [
    Extension(
        name="async_timer",
        sources=["async_timer.cpp"],
        include_dirs=[pybind11_include] + torch_include + [cuda_include],
        library_dirs=[cuda_lib],
        libraries=["cudart", "pthread"],
        # 修复：统一使用列表格式的编译参数（不区分cxx和nvcc）
        extra_compile_args=[
            "-O3", "-std=c++17", "-fPIC", "-Wall",
            "-I" + cuda_include  # 显式添加CUDA头文件路径
        ],
        extra_link_args=[
            "-L" + cuda_lib,  # 链接CUDA库路径
            "-lcudart", "-lpthread"
        ],
        language="c++"
    )
]

# 自定义编译类
class CustomBuildExt(build_ext):
    def build_extensions(self):
        # 确保编译器支持C++17
        for ext in self.extensions:
            if "-std=c++17" not in ext.extra_compile_args:
                ext.extra_compile_args.append("-std=c++17")
        build_ext.build_extensions(self)

# 执行安装
setup(
    name="async_timer",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False
)

