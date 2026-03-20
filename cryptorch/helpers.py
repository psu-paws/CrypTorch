import torch
from torch.fx.graph_module import GraphModule
import torch.fx as fx
from typing import Type, Any, Tuple, Iterable
from packaging.version import Version
from contextlib import contextmanager

from cryptorch.utils import fetch_attr
from torch._dispatch.python import enable_python_dispatcher

def matches_function_pattern(pattern: Iterable[Type], node: fx.Node):
    # Code modified from torch.fx.experimental.optimizations.py
    if len(node.args) == 0:
        return False
    nodes: Tuple[Any, fx.Node] = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != 'call_function':
            return False
        if not isinstance(current_node.target, torch._ops.OpOverload):
            return False
        if current_node.target is not expected_type:
            return False
    return True

def fuse(module: GraphModule) -> GraphModule:
    # Code modified from torch.fx.experimental.optimizations.py
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """
    patterns = [(torch.ops.aten.conv2d.default, torch.ops.aten.cudnn_batch_norm.default), (torch.ops.aten.conv2d.default, torch.ops.aten.batch_norm.default)]

    replacements = []
    for pattern in patterns:
        for node in module.graph.nodes:
            if matches_function_pattern(pattern, node):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                #conv = node.args[0].target
                #bn = node.target
                conv = node.args[0]
                bn = node
                '''
                if not bn.track_running_stats:
                    continue
                '''
                fuse_conv_bn(conv, bn, module)
                replacements.append((conv, bn))

                #print(fused_conv)
                #replace_node_module(node.args[0], modules, fused_conv)
                #node.replace_all_uses_with(node.args[0])
                #new_graph.erase_node(node)

    for conv, bn in replacements:
        with module.graph.inserting_before(conv):
            if Version(torch.__version__) < Version("2.6.0"):
                assert(len(bn.users) == 1)
                # bn always follwed by getitem. Remove getitem.
                users = []
                for n in bn.users:
                    users.append(n)
                for n in users:
                    n.replace_all_uses_with(conv)
                    module.graph.erase_node(n)
                module.graph.erase_node(bn)
            else:
                bn.replace_all_uses_with(conv)
                #print(bn, bn.args, bn.target)
                #exit(0)

                # Also remove bn params from the graph. For some reason, this is not done automatically by the dead code elimination.
                bn_params = bn.args[1:5]
                module.graph.erase_node(bn)
                for p in bn_params:
                    module.graph.erase_node(p)

            # TODO: pad with default values
            # If not using bias, use args
            args = list(conv.args)
            if len(args) < 3:
                args += [None,] * (3 - len(args))
            if args[2] is None:
               args[2] = module.graph.get_attr(args[1].target.rsplit(".", 1)[0] + ".bias")
               conv.args = tuple(args)
    #module = fx.GraphModule(module, module.graph)
    module.graph.eliminate_dead_code()
    module.graph.lint()
    module.recompile()
    return

# Adopted from torch.nn.utils.fusion.py
def fuse_conv_bn(conv, bn, module, transpose: bool = False):
    assert(conv.args[1].op == "get_attr")
    conv_w = fetch_attr(conv.args[1].target, module)
    if len(conv.args) >= 3 and conv.args[2] is not None:
        assert(conv.args[2].op == "get_attr")
        conv_b = fetch_attr(conv.args[2].target, module)
        #conv_b = fetch_attr(conv.args[2].target)
    else:
        conv_b = None
    assert(bn.args[1].op == "get_attr")
    assert(bn.args[2].op == "get_attr")
    assert(bn.args[3].op == "get_attr")
    assert(bn.args[4].op == "get_attr")
    bn_w = fetch_attr(bn.args[1].target, module)
    bn_b = fetch_attr(bn.args[2].target, module)
    bn_rm = fetch_attr(bn.args[3].target, module)
    bn_rv = fetch_attr(bn.args[4].target, module)
    bn_eps = bn.args[7]

    conv_weight_dtype = conv_w.dtype
    conv_bias_dtype = conv_b.dtype if conv_b is not None else conv_weight_dtype
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    if transpose:
        shape = [1, -1] + [1] * (len(conv_w.shape) - 2)
    else:
        shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)

    fused_conv_w = (conv_w * (bn_w * bn_var_rsqrt).reshape(shape)).to(dtype=conv_weight_dtype)
    fused_conv_b = ((conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b).to(dtype=conv_bias_dtype)

    module.register_parameter(f"{conv}_w_fused", torch.nn.Parameter(fused_conv_w))
    module.register_parameter(f"{conv}_b_fused", torch.nn.Parameter(fused_conv_b))
    conv.args[1].target = f"{conv}_w_fused"
    print(conv.args)
    if len(conv.args) >= 3 and conv.args[2] is not None:
        conv.args[2].target = f"{conv}_b_fused"
    else:
        with module.graph.inserting_before(conv):
            bias = module.graph.get_attr(f"{conv}_b_fused")
            new_args = list(conv.args)
            if len(new_args) < 3:
                new_args += [0] * (3 - len(new_args))
            new_args[2] = bias
            conv.args = tuple(new_args)

    return 

@contextmanager
def enable_custom_int_kernels():
    # Import this right before actually running MPC. Otherwise, this messes up with the export tracing.
    import cryptorch.custom_int_kernels_dispatcher
    with enable_python_dispatcher():
        yield
