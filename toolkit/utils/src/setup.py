from setuptools import setup, Extension

include_dirs = [
    # 添加包含 region.h 和 buffer.h 的路径
    r"E:\SiamProject\SiamTrackers-master\NanoTrack\toolkit\utils\src"
]

library_dirs = [
    # 添加包含需要链接的库的路径（如果有的话）
]

libraries = [
    # 添加需要链接的库的名称（不包括扩展名，如 .lib）（如果有的话）
]

extensions = [
    Extension("region", ["region.c"],
              include_dirs=include_dirs,
              library_dirs=library_dirs,
              libraries=libraries)
]

setup(
    name="Region Module",
    ext_modules=extensions
)

