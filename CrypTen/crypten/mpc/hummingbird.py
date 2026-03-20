import torch
import math
#from crypten.cuda.cuda_tensor import CUDALongTensor
from typing import Optional

hummingbird_msb = None

class HummingbirdOverride:
    def __init__(self, msb: Optional[None] = None):
        self.msb = msb
    
    def __enter__(self):
        self.old_msb = get_hummingbird_msb()
        set_hummingbird_msb(self.msb)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        set_hummingbird_msb(self.old_msb)

def set_hummingbird_msb(val):
    global hummingbird_msb
    hummingbird_msb = val

def get_hummingbird_msb():
    global hummingbird_msb
    return hummingbird_msb

def bitpack(t, bitwidth):
    t &= (2 ** bitwidth - 1)
    shape = t.shape
    t = t.flatten()
    total_size = t.shape[0]
    pack_ratio = math.floor(torch.iinfo(t.dtype).bits / bitwidth)
    size_after_bitpack = math.ceil(total_size / pack_ratio)
    padding_size = size_after_bitpack * pack_ratio - total_size
    #if isinstance(t, CUDALongTensor):
    #    t = CUDALongTensor.cat([t, CUDALongTensor(torch.zeros(padding_size, dtype=torch.int).to(t.device))])
    #    t = t.view(pack_ratio, -1)
    #    t2 = CUDALongTensor.stack([t[i] << (bitwidth * i) for i in range(pack_ratio)])
    #else:
    t = torch.cat([t, torch.zeros(padding_size, dtype=t.dtype, device=t.device)])
    t = t.view(pack_ratio, -1)
    t2 = torch.stack([t[i] << (bitwidth * i) for i in range(pack_ratio)])
    return t2.sum(dim=0, dtype=t.dtype), shape

def bitunpack(t, bitwidth, shape):
    pack_ratio = math.floor(torch.iinfo(t.dtype).bits / bitwidth)
    # replaced due incompatibility with Python < 3.8 
    total_size = 1
    for i in shape:
        total_size *= i
    # total_size = math.prod(shape)
    size_after_bitpack = math.ceil(total_size / pack_ratio)
    padding_size = size_after_bitpack * pack_ratio - total_size
    #if isinstance(t, CUDALongTensor):
    #    t2 = CUDALongTensor.stack([(t >> (bitwidth * i)) & ((2 ** bitwidth) - 1) for i in range(pack_ratio)])
    #else:
    t2 = torch.stack([(t >> (bitwidth * i)) & ((2 ** bitwidth) - 1) for i in range(pack_ratio)])
    if padding_size > 0:
        return t2.view(-1)[:-padding_size].view(shape)
    else:
        return t2.view(shape)
