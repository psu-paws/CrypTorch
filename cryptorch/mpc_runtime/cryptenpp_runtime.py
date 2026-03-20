from cryptorch.mpc_runtime.base_runtime import BaseRuntime
import os
from cryptorch.system_params import ring_size, ring_dtype, log_encoding_scale, encoding_scale, get_config_value

import torch
import crypten.communicator as comm
from crypten import crypten
from crypten.config import cfg
import tempfile
import yaml
import crypten.ring_size as rs
import crypten.mpc.primitives.beaver as beaver
from crypten.mpc.mpc import MPCTensor
from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor

# Implementation of the underlying MPC runtime for CrypTen++
class CrypTenPPRuntime(BaseRuntime):

    def init_runtime(self, rank, *args, **kwargs):
        os.environ['RANK'] = str(rank)
        os.environ['DISTRIBUTED_BACKEND'] = "gloo"
        
        backend_config = get_config_value("backend.config")
        backend_config_file_name = None
        if backend_config is not None:
            backend_config["communicator"]["comm_backend"] = "gloo"
            backend_config["communicator"]["verbose"] = True
            with tempfile.NamedTemporaryFile("w", delete=False) as f:
                yaml.dump(backend_config, f)
                backend_config_file_name = f.name
        else:
            cfg.communicator.comm_backend = "gloo"
            cfg.communicator.verbose = True
        crypten.init(config_file=backend_config_file_name)
        comm.get().reset_communication_stats()
        # Set ring size for CrypTen. This is a bit hacky..
        rs.set_ring_size(ring_size())
        return

    def get_comm_stats(self):
        return comm.get().get_communication_stats()

    def encode(self, x, scale):
        return getattr(x * scale, ring_dtype())()

    def encrypt(self, x, precision, src):
        return ArithmeticSharedTensor(x, src=src, precision=precision).share

    def decrypt(self, x, precision):
        x = MPCTensor.from_shares(x, precision=precision)
        return x.get_plain_text()

    def decrypt_sequence(self, x, precisions, owners):
        result = []
        for t, precision, owner in zip(x, precisions, owners):
            if owner < 0:
                owner = None
            decrypted_tensor = MPCTensor.from_shares(t, precision=precision).get_plain_text(dst=owner)
            if decrypted_tensor is None:
                decrypted_tensor = torch.zeros_like(t, dtype=torch.float32)
            result.append(decrypted_tensor)
        return result

    def ltz(self, x, *args, **kwargs):
        max_abs = kwargs["max_abs"]
        x = MPCTensor.from_shares(x, precision=log_encoding_scale())
        if max_abs is not None:
            msb = int(max_abs * 2 * encoding_scale()).bit_length() + 1
            # print(f"ltz: {max_abs=}, {msb=}")
            override_dict = {"functions.compare_msb": msb}
        else:
            override_dict = {}

        with cfg.temp_override(override_dict):
            result = x._ltz().share
        return result

    def div(self, x, y):
        x = MPCTensor.from_shares(x, precision=log_encoding_scale())
        return (x / y).share

    def conv2d(self, x, y, stride, padding):
        x = MPCTensor.from_shares(x, precision=log_encoding_scale())
        y = MPCTensor.from_shares(y, precision=log_encoding_scale())
        result = x.clone()
        kwargs = {"stride": 1, "padding": 0}
        if padding is not None:
            kwargs["padding"] = padding
        if stride is not None:
            kwargs["stride"] = stride
        # if dilation is not None:
        #     kwargs["dilation"] = dilation
        # kwargs["groups"] = groups
        result.share.set_(
            getattr(beaver, "conv2d")(x, y, **kwargs).share.data
        )
        return result.share

    def mul(self, x, y):
        x = MPCTensor.from_shares(x, precision=log_encoding_scale())
        y = MPCTensor.from_shares(y, precision=log_encoding_scale())
        result = x.clone()
        result.share.set_(
            getattr(beaver, "mul")(x, y).share.data
        )
        return result.share

    def mul_(self, x, y):
        x = MPCTensor.from_shares(x, precision=log_encoding_scale())
        y = MPCTensor.from_shares(y, precision=log_encoding_scale())
        result = x
        result.share.set_(
            getattr(beaver, "mul")(x, y).share.data
        )
        return result.share

    def square(self, x):
        x = MPCTensor.from_shares(x, precision=16)
        result = x.clone()
        result.share.set_(
            getattr(beaver, "square")(x).share.data
        )
        return result.share

    def square_(self, x):
        x = MPCTensor.from_shares(x, precision=16)
        result = x
        result.share.set_(
            getattr(beaver, "square")(x).share.data
        )
        return result.share

    def linear(self, x, y):
        x = MPCTensor.from_shares(x, precision=log_encoding_scale())
        y = MPCTensor.from_shares(y, precision=log_encoding_scale()).t()
        result = x.clone()
        result.share.set_(
            getattr(beaver, "matmul")(x, y).share.data
        )
        return result.share

    def matmul(self, x, y):
        x = MPCTensor.from_shares(x, precision=log_encoding_scale())
        y = MPCTensor.from_shares(y, precision=log_encoding_scale())
        result = x.clone()
        result.share.set_(
            getattr(beaver, "matmul")(x, y).share.data
        )
        return result.share

    def amax(self, x, dim, keepdim, *args, **kwargs):
        max_abs = kwargs["max_abs"]
        x = MPCTensor.from_shares(x, precision=log_encoding_scale())

        if max_abs is not None:
            msb = int(max_abs * 2 * encoding_scale()).bit_length() + 1
            override_dict = {"functions.compare_msb": msb}
        else:
            override_dict = {}

        # override_dict["functions.compare_lsb"] =  14
        with cfg.temp_override(override_dict):
            if dim is None or isinstance(dim, int):
                result = x.max(dim=dim, keepdim=keepdim, include_argmax=False)
            else:
                # cypten's max can only deal with 1 dim at a time
                for d in dim:
                    result = x.max(dim=d, keepdim=True, include_argmax=False)
                if not keepdim:
                    result = result.squeeze(dim=dim)
            return result.share
    
    def adaptive_avg_pool2d(self, x, output_size):
        x = MPCTensor.from_shares(x, precision=log_encoding_scale())
        result = x.adaptive_avg_pool2d(output_size)
        return result.share

    def max_pool2d(self, x, kernel_size, stride, padding, dilation, ceil_mode, *args, **kwargs):
        max_abs = kwargs["max_abs"]
        x = MPCTensor.from_shares(x, precision=log_encoding_scale())
        if max_abs is not None:
            msb = int(max_abs * 2 * encoding_scale()).bit_length() + 1
            override_dict = {"functions.compare_msb": msb}
            # print(f"max_pool: {max_abs=}, {msb=}")
        else:
            override_dict = {}
        
        if isinstance(kernel_size, list):
            kernel_size = tuple(kernel_size)
        if isinstance(stride, list):
            stride = tuple(stride)
        padding = padding or 0
        dilation = dilation or 1
        with cfg.temp_override(override_dict):
            result = x.max_pool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode, return_indices=False)
        return result.share
