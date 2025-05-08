# 导入 Python 的 setuptools 库中的 setup 函数，用于打包和安装模块
from setuptools import setup

# 从 PyTorch 的 C++ 扩展模块中导入 BuildExtension 和 CUDAExtension
# BuildExtension 用于扩展构建过程，CUDAExtension 用于定义包含 CUDA 代码的扩展模块
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 配置安装脚本
setup(
    # 指定该模块的名称，安装后可以通过 cx_cuda 进行标识（尽管导入时通常用子模块名）
    # 就是pip install cx_cuda 类似的名称
    name='cx_cuda',
    # 版本号，通常是模块的版本
    version='0.1.0',
    # ext_modules 是一个列表，其中每个元素定义一个要构建的扩展模块
    ext_modules=[
        CUDAExtension(
            # 构建后的 Python 模块名称，导入时使用，例如 import cx_kernels
            name='cx_kernels',
            # 指定该扩展模块的源文件，包括 C++ 和 CUDA 源文件
            sources=[
                'src/binding.cpp',         # C++ 绑定文件，通常用于 pybind11 接口定义
                'src/kernel/kernels.cu'    # CUDA 内核实现文件
            ]
        )
    ],
    # cmdclass 用于指定构建过程的自定义命令，此处使用 PyTorch 提供的 BuildExtension
    # 它自动处理 CUDA 文件编译和依赖管理
    cmdclass={
        'build_ext': BuildExtension
    }
)