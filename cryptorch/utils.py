import torch

def fetch_attr(attr_str, mod):
    #assert(n.op == "get_attr")
    attr_str = attr_str.split(".")

    obj = mod
    for attr in attr_str:
        obj = getattr(obj, attr)
    return obj

def set_attr(attr_str, val, mod):
    attr_str = attr_str.split('.')
    obj = mod
    for i, attr in enumerate(attr_str):
        if i == len(attr_str) - 1:
            setattr(obj, attr, val)
        else:
            if not hasattr(obj, attr):
                raise RuntimeError(f"Node referenced nonexistant target {'.'.join(attr_str[:i])}")
            obj = getattr(obj, attr)
    return

def get_src(n):
    for i, p in enumerate(n.meta["owner"]):
        if p:
            return i
    return None

def get_op_info(target):
    if isinstance(target, torch._ops.OpOverload):
        name = "{}.{}.{}".format(*target._schema.name.split("::"), target._overloadname)
    elif isinstance(target, torch._ops.OpOverloadPacket):
        name = "{}.{}.default".format(*target._qualified_op_name.split("::"))
    else:
        name = target.__name__

    if "." not in name:
        prefix = None
        op_name = name
        mode = None
    else:
        prefix = name.split(".")[0]
        op_name = name.split(".")[1]
        mode = name.split(".")[2]

    return prefix, op_name, mode

def is_secret(n):
    if not hasattr(n, "meta"):
        return False
    return False in n.meta["owner"]

# TODO: The cost is backend-specific, so this should be
# moved to the backend-specific part and be implemented separately
# for different backends. Currently using the CrypTen backend's GMW overhead.
def get_graph_cost(graph):
    cost = 0

    def get_numel(n):
        numel = 1
        shape = n.meta["tensor_meta"].shape
        for dim in shape:
            numel *= dim
        return numel

    for node in graph.nodes:
        if node.op == "call_function":
            prefix, op_name, mode = get_op_info(node.target)
            # TODO: right now I only measure the mul and comp cost, which are what are relevant for approx tuning.
            if op_name in ["mul", "mul_"]:
                assert(len(node.args) == 2)
                # TODO: We need to check if both inputs are secret, but it requires passing the secret info (which is possible); instead, we simply check if both inputs are tensor nodes for now.
                #if not sum([is_secret(arg) for arg in node.args]) == 2:
                if not sum([isinstance(arg, torch.fx.Node) for arg in node.args]) == 2:
                    continue
                numels = [get_numel(arg) for arg in node.args]
                print(numels)
                assert(numels[0] == numels[1])
                # Mul cost: 2N per element for N bits.
                # TODO: Again, this should be using the N from the backend and be moved to the backend-specific code.
                cost += numels[0] * 2 * 64
            elif op_name in ["lt", "gt", "ge", "le"]:
                # TODO: We need to check if the input is secret. We are not doing so, because it always is.
                #if not is_secret(node.args[0]):
                #    continue
                numel = get_numel(node.args[0])
                print(numel)
                # LTZ cost: 6N - 9 per element.
                cost += numel * (6 * 32 - 9)
                # cost += (numel * 2 * 64 - 1) # TODO: Fake cost for testing
    return cost

def get_inputs(m, intp):
    head = get_head(m.nodes_map)
    inputs, kwargs = intp.fetch_args_kwargs_from_env(m.nodes_map[head])
    return inputs
                        

def get_head(nodes_map):
    nodes = [k for k in nodes_map]

    for n in nodes:
        if n.op == "placeholder":
            continue
        is_head = True
        for arg in n.args:
            if hasattr(arg, "op") and arg.op != "placeholder":
                if arg in nodes:
                    is_head = False
                    continue
        if is_head:
            return n
    raise AssertionError() 


def _export_module(mod, inputs):
    #params = inspect.signature(mod.forward).parameters
    inputs_new = []
    mod.train()
    for x in inputs:
        #print(x, type(x))
        # I had this as a hack at one point, but don't remember exactly why. Maybe it is not needed anymore. Commenting this out as this does not work well for RNNs.
        '''
        if isinstance(x, torch.fx.immutable_collections.immutable_list):
            x = torch.Size(x)
        '''
        # TODO: This part breaks when an immutable is used as an input (as in nn.RNN).
        # This part fixes the problem, but the graph rewriting again breaks.
        if isinstance(x, torch.fx.immutable_collections.immutable_list):
            x = tuple(x)
        inputs_new.append(x)
    inputs = inputs_new
    # m = torch.export.export(mod, args=tuple(inputs)).module()
    return torch.export.export(mod, args=tuple(inputs)).module()

from torch.utils.data import Dataset
from numbers import Number
from collections.abc import Mapping
class FakeDataset(Dataset):
    def __init__(self, basis: Dataset):
        self.shapes = []
        for i in range(len(basis)):
            self.shapes.append(FakeDataset.convert_to_shapes(basis[i]))
    
    def __getitem__(self, index):
        return FakeDataset.convert_to_zero_tensors(self.shapes[index])
    
    def __len__(self):
        return len(self.shapes)
    
    @staticmethod
    def convert_to_shapes(obj):
        if isinstance(obj, torch.Tensor):
            return obj.shape
        elif isinstance(obj, Number):
            return 0
        elif isinstance(obj, tuple):
            return tuple((FakeDataset.convert_to_shapes(o) for o in obj))
        elif isinstance(obj, Mapping):
            return dict((key, FakeDataset.convert_to_shapes(value)) for key, value in obj.items())
        else:
            raise RuntimeError(f"Unsupported type: {type(obj)}")
    
    @staticmethod
    def convert_to_zero_tensors(obj):
        if isinstance(obj, torch.Size):
            return torch.zeros(obj)
        elif isinstance(obj, Number):
            return 0
        elif isinstance(obj, tuple):
            return tuple((FakeDataset.convert_to_zero_tensors(o) for o in obj))
        elif isinstance(obj, Mapping):
            return dict((key, FakeDataset.convert_to_zero_tensors(value)) for key, value in obj.items())
        else:
            raise RuntimeError(f"Unsupported type: {type(obj)}")

# def launch_mpc(module: torch.fx.GraphModule, )

# Helper functions for running mpc processes
from cryptorch.serialization import export_and_save, load_from
import cryptorch.system_params as sp

def _mpc_runner(f, config, serialized_module, rank, world_size, *args):
    module = load_from(serialized_module)
    sp.load_config(config)
    f(module, rank, world_size, *args)

def launch_mpc_processes(module, f, world_size=2, additional_arguments = None, *, init_method="tcp://127.0.0.1:8888"):
    # re-export
    import io
    serialized_module = io.BytesIO()
    
    
    export_and_save(serialized_module, module)
        
    # MPC eval
    import os
    import torch.multiprocessing as mp
    
    os.environ['WORLD_SIZE'] = str(world_size)

    mp.set_start_method("spawn", force=True)
    os.environ["RENDEZVOUS"] = init_method
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=_mpc_runner, args=(
            f,
            sp._system_config, 
            serialized_module, 
            rank, world_size) + tuple(additional_arguments[rank] if additional_arguments is not None else ()))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

