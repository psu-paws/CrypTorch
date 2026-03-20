from crypten.mpc.primitives.binary import BinarySharedTensor
import crypten.communicator as comm
from crypten.mpc.ptype import ptype as Ptype
from crypten.mpc.padding import flatten_and_pad, unflatten_and_unpad
from crypten.mpc.bit_packing import unmerge_bits, merge_bits
from crypten.mpc.hummingbird import get_hummingbird_msb, HummingbirdOverride
from crypten.config import cfg
# from crypten.cuda import CUDALongTensor
import crypten

import torch
from typing import Optional

import crypten.ring_size as rs

def ltz_fast(input, *, msb: Optional[int] = None, lsb: Optional[int] = None):
    """
    An optimized Less-than-zero (LTZ) implementation that only computes the sign bit in
    Arithmetic-to-Binary conversion to reduce bytes communicated
    """

    assert comm.get().get_world_size() == 2
    
    # print(f"{input.share.dtype=}")
    
    binary_tensors = BinarySharedTensor.stack(
            [
                BinarySharedTensor(input.share, src=i, device=input.device)
                for i in range(comm.get().get_world_size())
            ]
        )
    
    # print(f"{binary_tensors.share.dtype=}")
    
    result = input.clone()
    
    if lsb is None:
        lsb = 0

    if msb is None:
        msb = rs.get_ring_size()
    else:
        msb = min(msb, rs.get_ring_size())
        
    num_bits = msb - lsb
    
    binary_tensors = binary_tensors >> lsb

    # print(f"{num_bits=}, {lsb=}")
    # Override back to full-width to avoid shrinking iand
    with HummingbirdOverride(msb=None):
        if cfg.get("functions.compare") == "packed1":
            result._tensor = add_sign_only(binary_tensors[0], binary_tensors[1], num_bits=num_bits)
        elif cfg.get("functions.compare") == "packed2":
            result._tensor = add_sign_only2(binary_tensors[0], binary_tensors[1], num_bits=num_bits)
        else:
            raise RuntimeError(f'Unknown compare type: {cfg.get("functions.compare")}')
        result.ptype = Ptype.binary
    
    # TODO: REMOVE!
    # input_plain_text = input.get_plain_text()
    # if comm.get().get_rank() == 0:
    #     print(f"{input_plain_text=}")
    # real_result = input_plain_text < 0
    # result_plain_text = result.get_plain_text()
    # errors = real_result != result_plain_text
    # if comm.get().get_rank() == 0 and torch.count_nonzero(errors) > 0:
    #     print(f"input: {torch.masked_select(input_plain_text, errors)}")
    #     print(f"real_result: {torch.masked_select(real_result, errors)}")
    #     print(f"result: {torch.masked_select(result_plain_text, errors)}")
    
    return result


def add_sign_only(a, b, *, num_bits: Optional[int] = None):
    """
    Adds binary shared tensor a and b and return only the sign bit of the result in the LSB
    """
    num_bits_total = torch.iinfo(torch.long).bits
    
    if num_bits is None:
        num_bits = num_bits_total
    
    # sanity check
    # number of bits should be at least 1 and shouldn't be greater than total
    assert 0 < num_bits <= num_bits_total
    
    num_bits_next_power_of_2 = 2 ** ((num_bits - 1).bit_length())
    values_per_long = num_bits_total // num_bits_next_power_of_2
    
    # print(f"{num_bits=}, {num_bits_next_power_of_2=}, {values_per_long=}")
    
    value_mask = (1 << num_bits_next_power_of_2) - 1
    
    # print(f"{value_mask=:X}")
    
    shape = a.size()
    
    # compute propergate and generate
    propagate = a ^ b
    with HummingbirdOverride(msb=num_bits if num_bits != num_bits_total else None):
        generate = a & b
    propagate_msb = (propagate >> (num_bits - 1)) & 0x1
    propagate = propagate | -(1 << (num_bits - 1))
    generate = generate & ((1 << (num_bits - 1)) - 1)
    
    # pad to correct size
    propagate, padding = flatten_and_pad(propagate, num_bits_total * values_per_long)
    generate, _ = flatten_and_pad(generate, num_bits_total * values_per_long)
    
    # print(f"A{propagate.size()=}")
    # print(f"A{generate.size()=}")
    
    propagate = merge_bits(propagate, num_bits_next_power_of_2)
    generate = merge_bits(generate, num_bits_next_power_of_2)
    
    # print(f"B{propagate.size()=}")
    # print(f"B{generate.size()=}")
    
    propagate = propagate.view(num_bits_next_power_of_2, -1)
    generate = generate.view(num_bits_next_power_of_2, -1)
    
    # print(f"C{propagate.size()=}")
    # print(f"C{generate.size()=}")
    
    
    num_rounds = (num_bits - 1 - 1).bit_length()
    
    # print(f"{num_rounds=}")
    for round in range(num_rounds):
        step = 2 ** (round)
        mask = generate_mask(msb=num_bits_total, step=step * 2)
        # print(f"{hex(mask)=}")
        
        propagate_h = (propagate >> step) & mask
        propagate_l = propagate & mask
        
        generate_h = (generate >> step) & mask
        generate_l = generate & mask
        
        # interleaving bits to pack
        num_groups = num_bits_next_power_of_2 // 2 // step
        propagate_h = merge_bits_interleaved(propagate_h, num_groups, step)
        propagate_l = merge_bits_interleaved(propagate_l, num_groups, step)
        generate_h = merge_bits_interleaved(generate_h, num_groups, step)
        generate_l = merge_bits_interleaved(generate_l, num_groups, step)
        
        propagate, generate = PG_node(propagate_h, generate_h, propagate_l, generate_l)
        
        
        # print(f"{round}: {propagate.size()=}")
        # print(f"{round}: {generate.size()=}")
    
    
    # undo bit packing
    generate = unmerge_bits(generate, bits=1)
    
    generate = unflatten_and_unpad(generate, shape)
    
    result = propagate_msb ^ generate
    
    # print(f"{result.get_plain_text()=}")
    
    return result
    
def generate_mask(msb=64, step=2):
    result = 0
    for i in range(0, msb):
        if i % step < step // 2:
            result |= (1 << i)
    return result
def merge_bits_interleaved(t: BinarySharedTensor, num_groups, step):
    share = t.share
    share = share.view((num_groups, 2, -1))
    shift_amount = torch.tensor([0, step], device=share.device).type_as(share)
    shift_amount = shift_amount.view(1, 2, 1)
    share = shift_and_sum(share, shift_amount)
    t.share = share
    return t

def shift_and_sum(share, shift_amount):
    share = torch.bitwise_left_shift(share, shift_amount)
    share = torch.sum(share, dim=1)
    return share

def PG_node(propagate_h, generate_h, propagate_l, generate_l):
    # vectorized_and
    propagate_generate_l = BinarySharedTensor.stack([
        propagate_l,
        generate_l
    ])
    propagate_h_and_propagate_generate_l = propagate_h & propagate_generate_l
    propagate_out = propagate_h_and_propagate_generate_l[0]
    generate_out = propagate_h_and_propagate_generate_l[1] ^ generate_h
        
    return propagate_out, generate_out

def add_sign_only2(a, b, *, num_bits: Optional[int] = None):
    
    # print(f"1, {a.share.dtype=}")
    # print(f"1, {b.share.dtype=}")
    shape = a.size()
    a, _ = flatten_and_pad(a, rs.get_ring_size())
    b, _ = flatten_and_pad(b, rs.get_ring_size())
    
    # print(f"{a.size()=}")
    # print(f"{b.size()=}")
    
    # print(f"2, {a.share.dtype=}")
    # print(f"2, {b.share.dtype=}")
    
    a = bit_decomposition(a, bits=num_bits)
    b = bit_decomposition(b, bits=num_bits)
    
    # print(f"3, {a.share.dtype=}")
    # print(f"3, {b.share.dtype=}")
    
    # print(f"{a.size()=}")
    # print(f"{b.size()=}")
    
    # compute propergate and generate
    propagate = a[:-1] ^ b[:-1]
    propagate_msb = a[-1] ^ b[-1]
    generate = a[:-1] & b[:-1]
    
    
    # print(f"{propagate.size()=}")
    # print(f"{generate.size()=}")
    # print(f"{propagate_msb.size()=}")
    
    while (num_elements := propagate.size()[0]) != 1:
        is_odd = (num_elements & 1 != 0)
        high_indices = list(range(2 if is_odd else 1, num_elements, 2)) 
        low_indices = list(range(1 if is_odd else 0, num_elements, 2))
        
        # print(f"{high_indices=}")
        # print(f"{low_indices=}")
        
        propagate_h = propagate[high_indices]
        propagate_l = propagate[low_indices]
        generate_h = generate[high_indices]
        generate_l = generate[low_indices]
        
        # print(f"{propagate_h.size()=}")
        # print(f"{propagate_l.size()=}")
        # print(f"{generate_h.size()=}")
        # print(f"{generate_l.size()=}")
        
        if is_odd:
            propagate_left_over = propagate[0].view(1, -1)
            generate_left_over = generate[0].view(1, -1)
        
        propagate, generate = PG_node(propagate_h, generate_h, propagate_l, generate_l)
        
        if is_odd:
            propagate = BinarySharedTensor.cat([propagate_left_over, propagate])
            generate = BinarySharedTensor.cat([generate_left_over, generate])
        
        # print(f"{propagate.size()=}")
        # print(f"{generate.size()=}")
    
    # print(f"{propagate.size()=}")
    # print(f"{generate.size()=}")
    
    result = propagate_msb ^ generate.view(-1)
    
    result = unmerge_bits(result, bits=1)
    
    result = unflatten_and_unpad(result, shape)
    
    # print(f"{result.size()=}")
    
    return result

def bit_decomposition(input, bits=rs.get_ring_size()):
    tensors = []
    input_share = input.share
    # print(f"{input_share.dtype=}")
    input_share = input_share.view(rs.get_ring_size(), -1)
    shamt = torch.arange(0, rs.get_ring_size(), device=input_share.device).type_as(input_share).view(rs.get_ring_size(), 1)
    for i in range(bits):
        bit_extracted = (input_share >> i) & 1
        bit_extracted = torch.bitwise_left_shift(bit_extracted, shamt)
        # print(f"{bit_extracted.dtype=}")
        tensors.append(torch.sum(bit_extracted, dim=0, dtype=input_share.dtype))
    output = input.clone()
    output.share = torch.stack(tensors)
    # print(f"{output.share.dtype=}")
    return output
        
        
