import torch

from typing import Any
from cryptorch.utils import get_op_info

def hummingbird_tuning(module, test_inputs, get_x = None):
    class Interperter(torch.fx.Interpreter):
        def __init__(self, *args, **kwags):
            super().__init__(*args, **kwags)

        def run_node(self, node: torch.fx.Node) -> Any:
            
            if node.op == "call_function":
                prefix, op_name, mode = get_op_info(node.target)
                inputs, kwargs = super().fetch_args_kwargs_from_env(node)
                # print(f"{node}:{op_name}")
                # print(inputs[0])
                if (op_name == "lt" and node.args[1] == 0) or op_name == "amax" or op_name == "max_pool2d":
                    
                    if "max_abs" not in node.meta:
                        node.meta["max_abs"] = 0.0
                    
                    # print(node)
                    # if not torch.all(torch.isfinite(inputs[0])):
                    #     print("NAN")

                    input_values = torch.nan_to_num(inputs[0])
                    
                    input_abs = torch.abs(input_values)
                    # remove 
                    input_abs = torch.where(input_abs < 32768, input_abs, 0)
                    max_abs = torch.amax(input_abs)

                    # if not torch.all(torch.isfinite(input_vlaues)):
                    #     print(f"{node}")
                    #     print(input_vlaues)
                    # print(f"{node}: {max_abs}")
                    node.meta["max_abs"] = max(max_abs.item(), node.meta["max_abs"])


            result = super().run_node(node)
            return result

        def run(self, *args):
            return super().run(*args)
    
    intp = Interperter(module)
    with torch.no_grad():
        for i, batch in enumerate(test_inputs):
            x = (batch,) if get_x is None else get_x(batch)
            intp.run(*x)
