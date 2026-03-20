import torch
from typing import List, Optional
import cryptorch.custom_int_kernels as ck
from torch._dispatch.python import no_python_dispatcher


# Missing implementations for certain operators in torch.cuda.LongTensor.
# Cleanly overriding native functions were kind of hard for me, so I am using a dispatcher. I wonder if there's a better way
supported_rings = [torch.int64, torch.int32, torch.int16]
@torch.ops.aten.conv2d.default.py_impl(torch._C.DispatchKey.CUDA)
def f(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]=None, stride: List[int]=1, padding: List[int]=(0,0), dilation: list[int]=(1,1), groups=1) -> torch.Tensor:
    if x.dtype in supported_rings and w.dtype in supported_rings:
        return ck.conv2d(x, w, b, stride, padding, dilation, groups)
    else:
        with no_python_dispatcher():
            result = torch.ops.aten.conv2d.default(x, w, b, stride, padding, dilation, groups)
        return result

@torch.ops.aten.avg_pool2d.default.py_impl(torch._C.DispatchKey.CUDA)
def f(x: torch.Tensor, kernel_size: List[int], stride: Optional[List[int]]=None, padding: List[int]=0, ceil_mode: bool=False, count_include_pad: bool=True, divisor_override: Optional[int]=None) -> torch.Tensor:
    if x.dtype in supported_rings:
        return ck.avg_pool2d(x, kernel_size, stride, padding, ceil_mode)
    else:
        with no_python_dispatcher():
            result = torch.ops.aten.avg_pool2d.default(x, kernel_size, stride, padding, ceil_mode, divisor_override=divisor_override)
        return result

@torch.ops.aten.linear.default.py_impl(torch._C.DispatchKey.CUDA)
def f(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]=None):
    if x.dtype in supported_rings:
        assert(b is None)
        return ck.matmul(x, w.T)
    else:
        with no_python_dispatcher():
            result = torch.nn.functional.linear(x, w, b)
        return result

@torch.ops.aten.mm.default.py_impl(torch._C.DispatchKey.CUDA)
def f(x: torch.Tensor, y: torch.Tensor):
    if x.dtype in supported_rings:
        return ck.matmul(x, y)
    else:
        with no_python_dispatcher():
            result = torch.mm(x, y)
        return result

@torch.ops.aten.matmul.default.py_impl(torch._C.DispatchKey.CUDA)
def f(x: torch.Tensor, y: torch.Tensor):
    if x.dtype in supported_rings:
        return ck.matmul(x, y)
    else:
        with no_python_dispatcher():
            result = torch.matmul(x, y)
        return result
