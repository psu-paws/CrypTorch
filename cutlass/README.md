This repository contains an PyTorch extension that adds matrix multiply and convolution kernels in int64/int32 for CUDA GPUs based on the CUTLASS library. This is needed since by default PyTorch only comes with kernels for floating point and low-width integers such as int8.
# Build Instructions
## 1. Install nvidia-cuda-toolkit

The major and minor version should match your CUDA version which can be found in `nvidia-smi`. The major version should match the CUDA version your torch install is built with.

e.g., `conda install nvidia::cuda==12.4.0`

If you are using conda, double check that the `nvcc` version is what you expected, Nvidia somehow likes to install mismatched versions. If not, force install the correct version using the `cuda-nvcc` package. e.g., `conda install cuda-nvcc==12.4.131`

## 2. Install cutlass

* Clone cutlass
* Set environment variable CUTLASS_PATH to point at your cloned cutlass copy

* Run the following command to build and install this extension
```
python3 setup.py clean --all && python3 setup.py install
```

* Test by running `python3 test_gemm64.py`, it should print "Passed   True" at the end if everything is working.

### Common Issues
1. If there is an error saying your compiler is not modern enough, you may need to redefine CC and CXX to point to a more modern compiler.

2. If the test fails and the word Error is printed, double check the `nvcc` version matches your CUDA version displayed by `nvidia-smi` and the CUDA version your torch install is built with. If the correct version is install but the build system is using an incorrect version of `nvcc`, check that `PATH` is setup correctly by checking `which nvcc` and make sure CUDA_HOME is not set.
-------------------------------------------------

conv2d_int.cu and conv2d_int.h is partially derived from from [EzPC/GPU-MPC](https://github.com/mpc-msri/EzPC/tree/master/GPU-MPC).
