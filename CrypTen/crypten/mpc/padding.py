from crypten.mpc.primitives.binary import BinarySharedTensor
# from crypten.cuda.cuda_tensor import CUDALongTensor
import torch


def flatten_and_pad(t: BinarySharedTensor, pad_target: int=64):
    share = t.share
    size = share.nelement()
    share = torch.flatten(share)
    reminder = size % pad_target
    padding = 0 if reminder == 0 else pad_target - reminder
    # if isinstance(share, CUDALongTensor):
    #     share = CUDALongTensor.cat([share, CUDALongTensor(torch.zeros(padding, dtype=torch.int).to(t.device))])
    # else:
    share = torch.cat([share, torch.zeros(padding, dtype=share.dtype, device=share.device)])
    t.share = share
    return t, padding

def unflatten_and_unpad(t: BinarySharedTensor, target_shape: torch.Size):
    share = t.share
    share = torch.flatten(share)
    target_size = target_shape.numel()
    t.share = torch.reshape(share[:target_size], target_shape)
        
    return t