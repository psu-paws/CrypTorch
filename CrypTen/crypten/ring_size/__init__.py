import torch
_ring_size = 64

def set_ring_size(size):
    global _ring_size
    _ring_size = size

def get_ring_size():
    global _ring_size
    return _ring_size
    
def get_ring_dtype():
    global _ring_size
    if _ring_size == 64:
        return torch.int64
    elif _ring_size == 32:
        return torch.int32
    else:
        raise NotImplementedError()
