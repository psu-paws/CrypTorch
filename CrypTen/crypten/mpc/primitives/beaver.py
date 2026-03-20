#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten
import crypten.communicator as comm
import torch
from crypten.common.util import count_wraps
from crypten.config import cfg
from crypten.nn.module import time_per_op
import crypten.ring_size as rs

import time

class IgnoreEncodings:
    """Context Manager to ignore tensor encodings"""

    def __init__(self, list_of_tensors):
        self.list_of_tensors = list_of_tensors
        self.encodings_cache = [tensor.encoder.scale for tensor in list_of_tensors]

    def __enter__(self):
        for tensor in self.list_of_tensors:
            tensor.encoder._scale = 1

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for i, tensor in enumerate(self.list_of_tensors):
            tensor.encoder._scale = self.encodings_cache[i]


def __beaver_protocol(op, x, y, *args, **kwargs):
    """Performs Beaver protocol for additively secret-shared tensors x and y

    1. Obtain uniformly random sharings [a],[b] and [c] = [a * b]
    2. Additively hide [x] and [y] with appropriately sized [a] and [b]
    3. Open ([epsilon] = [x] - [a]) and ([delta] = [y] - [b])
    4. Return [z] = [c] + (epsilon * [b]) + ([a] * delta) + (epsilon * delta)
    """
    assert op in {
        "mul",
        "matmul",
        "conv1d",
        "conv2d",
        "conv_transpose1d",
        "conv_transpose2d",
    }
    start_t = time.time()
    if x.device != y.device:
        raise ValueError(f"x lives on device {x.device} but y on device {y.device}")

    provider = crypten.mpc.get_default_provider()
    a, b, c = provider.generate_additive_triple(
        x.size(), y.size(), op, device=x.device, *args, **kwargs
    )
    #print("Op 1 done", a.share.dtype, b.share.dtype, c.share.dtype)

    from .arithmetic import ArithmeticSharedTensor

    if cfg.mpc.active_security:
        """
        Reference: "Multiparty Computation from Somewhat Homomorphic Encryption"
        Link: https://eprint.iacr.org/2011/535.pdf
        """
        f, g, h = provider.generate_additive_triple(
            x.size(), y.size(), op, device=x.device, *args, **kwargs
        )

        t = ArithmeticSharedTensor.PRSS(a.size(), device=x.device)
        t_plain_text = t.get_plain_text()

        rho = (t_plain_text * a - f).get_plain_text()
        sigma = (b - g).get_plain_text()
        triples_check = t_plain_text * c - h - sigma * f - rho * g - rho * sigma
        triples_check = triples_check.get_plain_text()

        if torch.any(triples_check != 0):
            raise ValueError("Beaver Triples verification failed!")

    # Vectorized reveal to reduce rounds of communication
    with IgnoreEncodings([a, b, x, y]):
        epsilon, delta = ArithmeticSharedTensor.reveal_batch([x - a, y - b])

    # z = c + (a * delta) + (epsilon * b) + epsilon * delta
    c._tensor += getattr(torch, op)(epsilon, b._tensor, *args, **kwargs)
    #print("Op 2 done")
    c._tensor += getattr(torch, op)(a._tensor, delta, *args, **kwargs)
    #print("Op 3 done")
    c += getattr(torch, op)(epsilon, delta, *args, **kwargs)
    #print("Op 4 done")
    end_t = time.time()
    op += "_beaver"
    if op not in time_per_op:
        time_per_op[op] = 0.
    time_per_op[op] += end_t - start_t
    return c


def mul(x, y):
    return __beaver_protocol("mul", x, y)


def matmul(x, y):
    return __beaver_protocol("matmul", x, y)


def conv1d(x, y, **kwargs):
    return __beaver_protocol("conv1d", x, y, **kwargs)


def conv2d(x, y, **kwargs):
    return __beaver_protocol("conv2d", x, y, **kwargs)


def conv_transpose1d(x, y, **kwargs):
    return __beaver_protocol("conv_transpose1d", x, y, **kwargs)


def conv_transpose2d(x, y, **kwargs):
    return __beaver_protocol("conv_transpose2d", x, y, **kwargs)


def square(x):
    """Computes the square of `x` for additively secret-shared tensor `x`

    1. Obtain uniformly random sharings [r] and [r2] = [r * r]
    2. Additively hide [x] with appropriately sized [r]
    3. Open ([epsilon] = [x] - [r])
    4. Return z = [r2] + 2 * epsilon * [r] + epsilon ** 2
    """
    start_t = time.time()
    provider = crypten.mpc.get_default_provider()
    r, r2 = provider.square(x.size(), device=x.device)

    with IgnoreEncodings([x, r]):
        epsilon = (x - r).reveal()
    y = r2 + 2 * r * epsilon + epsilon * epsilon
    end_t = time.time()
    if "square" not in time_per_op:
        time_per_op["square"] = 0.
    time_per_op["square"] += end_t - start_t
    return y


def wraps(x):
    """Privately computes the number of wraparounds for a set a shares

    To do so, we note that:
        [theta_x] = theta_z + [beta_xr] - [theta_r] - [eta_xr]

    Where [theta_i] is the wraps for a variable i
          [beta_ij] is the differential wraps for variables i and j
          [eta_ij]  is the plaintext wraps for variables i and j

    Note: Since [eta_xr] = 0 with probability 1 - |x| / Q for modulus Q, we
    can make the assumption that [eta_xr] = 0 with high probability.
    """
    provider = crypten.mpc.get_default_provider()
    r, theta_r = provider.wrap_rng(x.size(), device=x.device)
    beta_xr = theta_r.clone()
    beta_xr._tensor = count_wraps([x._tensor, r._tensor])

    with IgnoreEncodings([x, r]):
        z = x + r
    theta_z = comm.get().gather(z._tensor, 0)
    theta_x = beta_xr - theta_r

    # TODO: Incorporate eta_xr
    if x.rank == 0:
        theta_z = count_wraps(theta_z)
        theta_x._tensor += theta_z
    return theta_x


def truncate(x, y):
    """Protocol to divide an ArithmeticSharedTensor `x` by a constant integer `y`"""
    wrap_count = wraps(x)
    x.share = x.share.div_(y, rounding_mode="trunc")
    # NOTE: The multiplication here must be split into two parts
    # to avoid long out-of-bounds when y <= 2 since (2 ** 63) is
    # larger than the largest long integer.
    correction = wrap_count * 4 * (int(2**62) // y)
    x.share -= correction.share
    return x


def truncate_aby3(x, y):
    """Protocol to divide an ArithmeticSharedTensor `x` by a constant integer `y`
    This implements what was proposed in the ABY3 paper. Note: There is a paper claiming that this method is insecure (I don't know how significant the vulnerability is). Still implementing for comparison with other frameworks using it.
    """
    provider = crypten.mpc.get_default_provider()
    r, r_ = provider.generate_truncation_rng(y, x.size(), device=x.device)
    with IgnoreEncodings([x, r_]):
        x_r_ = (x - r_).reveal()
    r += x_r_.__floordiv__(y)
    x.share = r.share
    return x


def truncate_reduced_slack(x_original, y):
    # Implementation based on Figure 2 of Truncation Untangled
    # https://petsymposium.org/popets/2025/popets-2025-0135.pdf
    
    # print("AAAAAAAAA")
    
    assert y.bit_count() == 1
    t = y.bit_length() - 1
    with IgnoreEncodings([x_original,]):
        # print(f"{x_original.get_plain_text()=}")
        x = x_original + 2**(rs.get_ring_size() - 2)
        # x = x_original
        # print(f"{x.get_plain_text()=}")
        provider = crypten.mpc.get_default_provider()
        r, rMSB, rUpper = provider.RSST_rng(x.size(), t, device=x.device)
        
        # print(f"{r.get_plain_text()=}")
        # print(f"{rMSB.get_plain_text()=}")
        # print(f"{rUpper.get_plain_text()=}")
        
        c = (x + r).reveal()
        # print(f"{c=}")
        # c_mask = (2 ** (64 - t)) - 1
        c_prime = (c >> t) % (2 ** (rs.get_ring_size() - t -1))
        
        # print(f"{c=}")
        # print(f"{c_prime=}")
        # print(f"{c_mask=}")
        
        c_msb = c >> 63 & 1
        
        not_rMSB = 1 - rMSB
        # print(f"{(rMSB - not_rMSB)=}")
        # print(f"{c_msb=}")
        b = rMSB + (not_rMSB - rMSB) * c_msb
        # print(f"{b.get_plain_text()=}")
        # print(f"{c_prime=}")
        # print(f"{(-rUpper + c_prime).get_plain_text()=}")
        y = -rUpper + c_prime + b * 2 ** (rs.get_ring_size() - t - 1)
        # print(f"{y.get_plain_text()=}")
        y = y - 2 ** (rs.get_ring_size() - t - 2)
        # print(f"POST: {y.get_plain_text()=}")
        x_original.share = y.share
        
    # print(f"{x_original.encoder._scale=}")
    # print(f"{x_original.get_plain_text()=}")
    return x_original

def AND(x, y):
    """
    Performs Beaver protocol for binary secret-shared tensors x and y

    1. Obtain uniformly random sharings [a],[b] and [c] = [a & b]
    2. XOR hide [x] and [y] with appropriately sized [a] and [b]
    3. Open ([epsilon] = [x] ^ [a]) and ([delta] = [y] ^ [b])
    4. Return [c] ^ (epsilon & [b]) ^ ([a] & delta) ^ (epsilon & delta)
    """
    from .binary import BinarySharedTensor
    provider = crypten.mpc.get_default_provider()
    a, b, c = provider.generate_binary_triple(x.size(), y.size(), device=x.device)
    
    # print(f"{a.share.dtype=}")
    # print(f"{b.share.dtype=}")
    # print(f"{c.share.dtype=}")
    
    # print(f"{x.share.dtype=}")
    # print(f"{(x ^ a).share.dtype=}")
    
    # Stack to vectorize reveal
    eps_del = BinarySharedTensor.reveal_batch([x ^ a, y ^ b])
    epsilon = eps_del[0]
    delta = eps_del[1]
    
    # print(f"{epsilon.dtype}")
    # print(f"{delta.dtype}")
    

    return (b & epsilon) ^ (a & delta) ^ (epsilon & delta) ^ c


def B2A_single_bit(xB):
    """Converts a single-bit BinarySharedTensor xB into an
        ArithmeticSharedTensor. This is done by:

    1. Generate ArithmeticSharedTensor [rA] and BinarySharedTensor =rB= with
        a common 1-bit value r.
    2. Hide xB with rB and open xB ^ rB
    3. If xB ^ rB = 0, then return [rA], otherwise return 1 - [rA]
        Note: This is an arithmetic xor of a single bit.
    """
    if comm.get().get_world_size() < 2:
        from .arithmetic import ArithmeticSharedTensor

        return ArithmeticSharedTensor(xB._tensor, precision=0, src=0)

    provider = crypten.mpc.get_default_provider()
    rA, rB = provider.B2A_rng(xB.size(), device=xB.device)

    z = (xB ^ rB).reveal(singlebit=True)
    rA = rA * (1 - 2 * z) + z
    return rA
