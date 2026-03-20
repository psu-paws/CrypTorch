from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
  
# Get cutlass source directory
import os

cutlass_path = os.environ["CUTLASS_PATH"]

setup(
        name='int64_kernels',
        ext_modules=[
            CUDAExtension('conv2d_int', [
                'conv2d_int_wrapper.cpp', 
                'conv2d_int.cu',
            ],
                          extra_compile_args={
                              'nvcc' : [
                                  '-I%s' % cutlass_path + "/include/",
                                  '-I%s' % cutlass_path + "/tools/util/include/",
                                  '-arch=compute_86',
                                  '-code=compute_86',
                                  '-L/usr/local/cuda/lib64',
                                  '-lcuda',
                                  '-lcudart',
                                  '-lcublas',
                                  '-lcurand',
                              ],
                              'cxx' : ['-Ofast',
                                       '-I%s' % cutlass_path,
                                       '-I%s' % cutlass_path + "/include/"
                                       ]
                          }
            ),
            CUDAExtension('gemm_int', [
                'gemm_int_wrapper.cpp', 
                'gemm_int.cu',
            ],
                          extra_compile_args={
                              'nvcc' : [
                                  '-I%s' % cutlass_path + "/include/",
                                  '-I%s' % cutlass_path + "/tools/util/include/",
                                  '-arch=compute_86',
                                  '-code=compute_86',
                                  '-L/usr/local/cuda/lib64',
                                  '-lcuda',
                                  '-lcudart',
                                  '-lcublas',
                                  '-lcurand',
                              ],
                              'cxx' : ['-Ofast',
                                       '-I%s' % cutlass_path,
                                       '-I%s' % cutlass_path + "/include/"
                                       ]
                          }
            ),
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
