from crypten.mpc.primitives.binary import BinarySharedTensor
from crypten.mpc.padding import flatten_and_pad
import torch

def merge_bits(t: BinarySharedTensor, bits=1):
    share = t.share
    
    total_bits = torch.iinfo(share.dtype).bits
    
    mask = -1 if bits == total_bits else (1 << bits) - 1
    
    share = share.view(total_bits // bits, -1)
    share = share & mask
    shift_amount = torch.arange(0, total_bits, step=bits, device=share.device).type_as(share)
    shift_amount = shift_amount.view(-1, 1)
    share = torch.sum(torch.bitwise_left_shift(share, shift_amount), dim=0)
    t.share = share
    # print(share.shape)
    return t


def unmerge_bits(t: BinarySharedTensor, bits=1):
    share = t.share
    
    total_bits = torch.iinfo(share.dtype).bits
    
    mask = -1 if bits == total_bits else (1 << bits) - 1
    
    shift_amount = torch.arange(0, total_bits, step=bits, device=share.device).type_as(share)
    shift_amount = shift_amount.view(-1, 1)
    share = torch.bitwise_right_shift(share, shift_amount)
    share = share & mask
    t.share = share
    # print(share.shape)
    return t