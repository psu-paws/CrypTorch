import torch
import conv2d_int
import gemm_int
import math
from cryptorch.system_params import ring_size, ring_dtype

# avg_pool2d copied from CrypTen's implementation with modifications.
def avg_pool2d(x, kernel_size, stride=None, padding=0, ceil_mode=False):
    if stride is None:
        stride = kernel_size
    nb = 3
    bks = math.ceil(ring_size() / nb)

    def __encode_as_fp64(x):
        x_block = torch.stack(
            [(x >> (bks * i)) & (2**bks - 1) for i in range(nb)]
        )
        return x_block.double()

    x_encoded = __encode_as_fp64(x).data

    bs, c, h, w = x.shape
    x_encoded = x_encoded.reshape(nb * bs, c, h, w)

    z_encoded = torch.nn.functional.avg_pool2d(
        x_encoded, kernel_size, divisor_override=1, stride=stride, padding=padding, ceil_mode=ceil_mode
    )

    z_enc = getattr(z_encoded.reshape(nb, bs, *z_encoded.shape[1:]), ring_dtype())()
    z = torch.zeros(
        (nb, bs, *z_encoded.shape[1:]), device=x.device, dtype=getattr(torch, ring_dtype())
    )
    z += z_enc << torch.tensor([bks * i for i in range(nb)], device=x.device).view(
        nb, 1, 1, 1, 1
    )
    z = z.sum(0, dtype=getattr(torch, ring_dtype()))

    if isinstance(kernel_size, (int, float)):
        pool_size = kernel_size**2
    else:
        pool_size = kernel_size[0] * kernel_size[1]

    z = torch.div(z, pool_size, rounding_mode="trunc")

    return z


def conv2d(input, filter, bias=None, stride=1, padding=0, dilation=1, groups=1):
    assert(bias is None)
    
    # TODO: Hack for group conv
    if groups != 1:
        assert False
        # input = CUDALongTensor(data=input)
        # filter = CUDALongTensor(data=filter)
        # result = CUDALongTensor.conv2d(input, filter, stride=stride, padding=padding, dilation=dilation, groups=groups)
        # return result.tensor()
    
    BS, CI, H, W = tuple(input.shape)
    CO, CI_, FH, FW = tuple(filter.shape)
    assert(CI == CI_ * groups)
    assert(CO % groups == 0)

    if isinstance(stride, (tuple, list)):
        assert(stride[0] == stride[1])
        stride = stride[0]
    if isinstance(padding, (tuple, list)):
        assert(padding[0] == padding[1])
        padding = padding[0]
    if isinstance(dilation, (tuple, list)):
        assert(dilation[0] == dilation[1])
        dilation = dilation[1]

    OH = ((H - FH + 2 * padding) // stride) + 1
    OW = ((W - FW + 2 * padding) // stride) + 1

    out = torch.zeros([BS, OH, OW, CO], dtype=torch.int32 if input.dtype == torch.int16 else input.dtype, device=input.device)
    # Cutlass uses NHWC
    input_r = input.permute(0, 2, 3, 1).contiguous()
    filter_r = filter.permute(0, 2, 3, 1).contiguous()
    #print(x.dtype, y.dtype, out.dtype)

    if input.dtype == torch.int64:
        conv2d_int.conv2d64(input_r, filter_r, out, BS, CI, H, W, CO, FH, FW, stride, padding)
    elif input.dtype == torch.int32:
        conv2d_int.conv2d32(input_r, filter_r, out, BS, CI, H, W, CO, FH, FW, stride, padding)
    elif input.dtype == torch.int16:
        conv2d_int.conv2d32(
            input_r.to(torch.int32, memory_format=torch.contiguous_format),
            filter_r.to(torch.int32, memory_format=torch.contiguous_format),
            out,
            BS, CI, H, W, CO, FH, FW, stride, padding
        )
        out = out.to(torch.int16)
    else:
        raise NotImplementedError()

    out = out.permute(0, 3, 1, 2)
    return out

def matmul(x, y):
    if len(y.shape) == 2:
        # None-batched gemm (linear)
        x_shape = x.shape
        y_shape = y.shape
        assert(len(x_shape) in [2, 3])
        assert(len(y_shape) == 2)
        if len(x_shape) == 2:
            M = x_shape[0]
            K = x_shape[1]
            N = y_shape[1]
            out = torch.zeros([M, N], dtype=torch.int32 if x.dtype == torch.int16 else x.dtype, device=x.device)
            if x.dtype == torch.int64:
                gemm_int.matmul64(x.contiguous(), y.contiguous(), out, M, K, N, 1)
            elif x.dtype == torch.int32:
                gemm_int.matmul32(x.contiguous(), y.contiguous(), out, M, K, N, 1)
            elif x.dtype == torch.int16:
                gemm_int.matmul32(
                    x.to(torch.int32, memory_format=torch.contiguous_format),
                    y.to(torch.int32, memory_format=torch.contiguous_format),
                    out,
                    M, K, N ,1
                    )
                out = out.to(torch.int16)
            else:
                raise NotImplementedError()
        else:
            BS = x_shape[0]
            M = x_shape[1]
            K = x_shape[2]
            N = y_shape[1]
            out = torch.zeros([BS * M, N], dtype=torch.int32 if x.dtype == torch.int16 else x.dtype, device=x.device)
            if x.dtype == torch.int64:
                gemm_int.matmul64(x.contiguous(), y.contiguous(), out, BS * M, K, N, 1)
            elif x.dtype == torch.int32:
                gemm_int.matmul32(x.contiguous(), y.contiguous(), out, BS * M, K, N, 1)
            elif x.dtype == torch.int16:
                gemm_int.matmul32(
                    x.to(torch.int32, memory_format=torch.contiguous_format),
                    y.to(torch.int32, memory_format=torch.contiguous_format),
                    out,
                    BS * M, K, N ,1
                    )
                out = out.to(torch.int16)
            else:
                raise NotImplementedError()
            out = out.reshape(BS, M, N)
        return out
    else:
        # Batched gemm (bmm)
        x_shape = x.shape
        y_shape = y.shape
        assert(len(x_shape) == len(y_shape))
        M, K = x_shape[-2], x_shape[-1]
        N = y_shape[-1]
        assert(K == y_shape[-2])
        x_t = x.reshape(-1, M, K)
        y_t = y.reshape(-1, K, N)
        bs = x_t.shape[0]
        assert(bs == y_t.shape[0])

        out = torch.zeros([bs, M, N], dtype=torch.long).to(x.device)
        gemm_int.matmul64(x_t.contiguous(), y_t.contiguous(), out, M, K, N, bs)
        out = out.reshape(*x_shape[:-2], M, N)
        return out
