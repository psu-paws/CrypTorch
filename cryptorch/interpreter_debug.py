# mypy: ignore-errors

import torch
import torch.fx
import numpy as np

from torch.fx.node import Node
from typing import Any
from torch.fx._compatibility import compatibility

import contextlib
from cryptorch.system_params import ring_dtype

@compatibility(is_backward_compatible=True)
class Executor(torch.fx.Interpreter):
    def __init__(self, mod):
        super().__init__(mod)

    def run_node(self, n : Node) -> Any:
        # Get any inputs being passed through a higher-level module
        #if self.rank == 0:
        #    print(f"Run {n} {n.op} {n.target}")
        inputs, kwargs = super().fetch_args_kwargs_from_env(n)
       
        print(f"Run {n} {n.op} {n.target}")
        print("Inputs", [[(type(y), y.dtype) for y in x] if isinstance(x, tuple) else ((type(x), x.dtype) if hasattr(x, "shape") else (type(x), x)) for x in inputs])
        # if self.rank == 0:
        #     print(f"Run {n} {n.op} {n.target} {n.meta['owner']}")
        #     print("Inputs", [[(type(y), y.shape) for y in x] if isinstance(x, tuple) else ((type(x), x.size()) if (torch.is_tensor(x) or self.backend.is_encrypted(x)) else (type(x), x)) for x in inputs])
        # print(n)
        
        result = super().run_node(n)
        #plain_res = result
        #try:
        #    max_val = plain_res.max()
        #    min_val = plain_res.min()
        #except:
        #    max_val = None
        #    min_val = None
        #print(f"{max_val=} {min_val=} {plain_res=}")

        # ========================== Printing for non-MPC =====================
        # Avoid the float being "inf" due to becoming too large (this will be an overflow for MPC).
        #if isinstance(result, torch.Tensor):
        #    result = result.nan_to_num(nan=0, posinf=2 ** 63 - 1, neginf=-2 ** 63)
        #if hasattr(n, "meta") and "encoding_scale" in n.meta and n.meta["encoding_scale"] is not None:
        #    print(f"Scaled with factor={n.meta['encoding_scale']}")
        #    plain_res = result / n.meta["encoding_scale"]
        #else:
        #    plain_res = result
        #try:
        #    max_val = plain_res.max()
        #    min_val = plain_res.min()
        #except:
        #    max_val = None
        #    min_val = None
        #print(f"{max_val=} {min_val=} {plain_res=}")

        #if n.op == "output":
        #    result = plain_res

        # ========================== Printing for MPC =====================
        if hasattr(result, "dtype"):
            print("RES:", n, n.target, result.dtype)
            if result.dtype in [torch.int, torch.long]:
                assert(result.dtype == getattr(torch, ring_dtype))
        #if hasattr(result, "dtype") and result.dtype == torch.int64:
        #    plain_res = torch.ops.cryptorch.decrypt(result, 16)
        #    try:
        #        max_val = plain_res.max()
        #        min_val = plain_res.min()
        #    except:
        #        max_val = None
        #        min_val = None
        #    print(f"{max_val=} {min_val=} {plain_res=}")
        return result
    
    def run(self, *args, log_to_file=False, log_comm_stats=False):
        # self.log_to_file = log_to_file
        with contextlib.ExitStack() as stack:
            self.log_to_file = log_to_file
            if log_to_file and self.rank == 0:
                self.log_file = stack.enter_context(open("log.txt", "w"))
            
            if log_comm_stats and self.rank == 0:
                self.log_comm_stats = True
                self.comm_log_file = stack.enter_context(open("comm.log", "w"))
            else:
                self.log_comm_stats = False
            return super().run(*args)
