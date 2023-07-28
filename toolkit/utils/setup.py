from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

include_dirs = [
    "utils.src",  # 包含 region.h 和 buffer.h 的路径
    np.get_include()  # 添加 NumPy 头文件路径，如果你在 region.pyx 中使用了 NumPy
]

extensions = [
    Extension("region",
              sources=["region.pyx", "src/region.c"],
              include_dirs=include_dirs)
]

setup(
    name="Region Module",
    ext_modules=cythonize(extensions)
)
