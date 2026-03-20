import torch
from typing import List, Optional, Sequence

# Skeleton code to call the relevant MPC runtime functions
_runtime = None
def init_runtime(runtime, rank, *args, **kwargs):
    global _runtime
    _runtime = runtime
    runtime.init_runtime(rank, *args, **kwargs)

def get_comm_stats():
    global _runtime
    return _runtime.get_comm_stats()

@torch.library.custom_op("cryptorch::encode", mutates_args={})
def f(x: torch.Tensor, scale: int) -> torch.Tensor:
    global _runtime
    return _runtime.encode(x, scale)
@torch.library.register_fake("cryptorch::encode")
def f(x, scale):
    return torch.empty_like(x)

@torch.library.custom_op("cryptorch::encrypt", mutates_args={})
def f(x: torch.Tensor, precision: int, src: int) -> torch.Tensor:
    global _runtime
    return _runtime.encrypt(x, precision, src)
@torch.library.register_fake("cryptorch::encrypt")
def f(x, precision, src):
    return torch.empty_like(x)

@torch.library.custom_op("cryptorch::decrypt", mutates_args={})
def f(x: torch.Tensor, precision: int) -> torch.Tensor:
    global _runtime
    return _runtime.decrypt(x, precision)
@torch.library.register_fake("cryptorch::decrypt")
def f(x, precision):
    return torch.empty_like(x)

@torch.library.custom_op("cryptorch::decrypt_sequence", mutates_args={})
def f(x: Sequence[torch.Tensor], precisions: Sequence[int], owners: Sequence[int]) -> list[torch.Tensor]:
    global _runtime
    return _runtime.decrypt_sequence(x, precisions, owners)
@torch.library.register_fake("cryptorch::decrypt_sequence")
def f(x, precision, owners):
    return torch.empty_like(x)

@torch.library.custom_op("cryptorch::dec_fake", mutates_args={})
def f(x: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [t.clone() for t in x]
@torch.library.register_fake("cryptorch::dec_fake")
def f(x):
    return x

@torch.library.custom_op("cryptorch::ltz", mutates_args={})
def f(x: torch.Tensor, *, max_abs: Optional[float]=None) -> torch.Tensor:
    global _runtime
    return _runtime.ltz(x, max_abs=max_abs)
@torch.library.register_fake("cryptorch::ltz")
def f(x, *, max_abs: Optional[float]=None):
    return torch.empty_like(x).bool()

@torch.library.custom_op("cryptorch::conv2d", mutates_args={})
def f(x: torch.Tensor, y: torch.Tensor, bias: Optional[torch.Tensor], stride: Optional[List[int]]=None, padding: Optional[List[int]]=None) -> torch.Tensor:
    assert(bias is None)
    global _runtime
    return _runtime.conv2d(x, y, stride, padding)
@torch.library.register_fake("cryptorch::conv2d")
def f(x: torch.Tensor, y: torch.Tensor, stride: Optional[List[int]], padding: Optional[List[int]]) -> torch.Tensor:
    return torch.nn.functional.conv2d(x, y, stride=stride, padding=padding)

@torch.library.custom_op("cryptorch::mul", mutates_args={})
def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    global _runtime
    return _runtime.mul(x, y)
@torch.library.register_fake("cryptorch::mul")
def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * y

@torch.library.custom_op("cryptorch::square", mutates_args={})
def f(x: torch.Tensor) -> torch.Tensor:
    global _runtime
    return _runtime.square(x)
@torch.library.register_fake("cryptorch::square")
def f(x: torch.Tensor) -> torch.Tensor:
    return x ** 2

@torch.library.custom_op("cryptorch::square_", mutates_args={})
def f(x: torch.Tensor) -> torch.Tensor:
    global _runtime
    return _runtime.square_(x)
@torch.library.register_fake("cryptorch::square_")
def f(x: torch.Tensor) -> torch.Tensor:
    return x ** 2

@torch.library.custom_op("cryptorch::linear", mutates_args={})
def f(x: torch.Tensor, y: torch.Tensor, b: Optional[torch.Tensor]=None) -> torch.Tensor:
    global _runtime
    assert(b is None)
    return _runtime.linear(x, y)
@torch.library.register_fake("cryptorch::linear")
def f(x: torch.Tensor, y: torch.Tensor, b: Optional[torch.Tensor]=None) -> torch.Tensor:
    return torch.nn.functional.linear(x, y)

@torch.library.custom_op("cryptorch::matmul", mutates_args={})
def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    global _runtime
    return _runtime.matmul(x, y)
@torch.library.register_fake("cryptorch::matmul")
def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, y)

@torch.library.custom_op("cryptorch::mul_", mutates_args={"x"})
def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    global _runtime
    return _runtime.mul_(x, y)
@torch.library.register_fake("cryptorch::mul_")
def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.mul(y)

@torch.library.custom_op("cryptorch::amax", mutates_args={})
def f(x: torch.Tensor, dim: List[int], keepdim: bool, *, max_abs: Optional[float]=None) -> torch.Tensor:
    global _runtime
    return _runtime.amax(x, dim, keepdim, max_abs=max_abs)
@torch.library.register_fake("cryptorch::amax")
def f(x: torch.Tensor, dim: List[int], keepdim: bool, *, max_abs: Optional[float]=None) -> torch.Tensor:
    return torch.amax(x, dim, keepdim)

@torch.library.custom_op("cryptorch::div", mutates_args={})
def f(x: torch.Tensor, y: int) -> torch.Tensor:
    global _runtime
    return _runtime.div(x, y)
@torch.library.register_fake("cryptorch::div")
def f(x: torch.Tensor, y: int) -> torch.Tensor:
    return x.div(y, rounding_mode="trunc")

@torch.library.custom_op("cryptorch::adaptive_avg_pool2d", mutates_args={})
def f(x: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    global _runtime
    return _runtime.adaptive_avg_pool2d(x, output_size)
@torch.library.register_fake("cryptorch::adaptive_avg_pool2d")
def f(x: torch.Tensor, output_size: List[int]) -> torch.Tensor:
    return x.adaptive_avg_pool2d(output_size)

@torch.library.custom_op("cryptorch::max_pool2d", mutates_args={})
def f(x: torch.Tensor, kernel_size: List[int], stride: Optional[List[int]]=None, padding: Optional[List[int]]=None, dilation: Optional[List[int]]=None, ceil_mode: bool=False, *, max_abs: Optional[float]=None) -> torch.Tensor:
    global _runtime
    return _runtime.max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode, max_abs=max_abs)
@torch.library.register_fake("cryptorch::max_pool2d")
def f(x: torch.Tensor, kernel_size: List[int], stride: Optional[List[int]]=None, padding: Optional[List[int]]=None, dilation: Optional[List[int]]=None, ceil_mode: bool=False) -> torch.Tensor:
    return x.max_pool2d(kernel_size, stride, padding, dilation, ceil_mode)
