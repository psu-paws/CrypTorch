import torch
from cryptorch.system_params import log_encoding_scale, encoding_scale, ring_dtype
from cryptorch.utils import get_op_info, is_secret, get_src, fetch_attr
# This is not needed for inference, but putting in case of training.
#from cryptorch.cryptorch_tensor import CrypTorchTensor

# The pass manager is not powerful enough to do transformations I want, so I am writing a separate passes (not using the pass manager style).

# Bitwise not is rsub (1 - x), so it is considered additive
def is_additive(name):
    if name in ["add", "sub", "rsub", "add_", "sub_", "rsub_", "bitwise_not"]:
        return True
    else:
        return False

def copy_passes_meta(src, dest):
    if "passes" in src.meta:
        dest.meta["passes"] = src.meta["passes"]

# TODO: Merge this with the code in backend/base.py
# TODO: There is no good way to check which args are data vs. params (e.g., stride, padding),
# so we use a bit of a hack.
def get_compute_input_idxs(target, arg_num):
    # Return arg idxs for values that are actually used
    # in computation (e.g., tensors or scalars that are not stride, padding, ...)
    prefix, op_name, mode = get_op_info(target)
    schema = target._schema
    args = schema.arguments
    idxs = []
    # Using arg num because we only care about args that this op really has
    for i in range(arg_num):
        arg = args[i]
        ty = arg.type
        if isinstance(ty, (torch.TensorType, torch.NumberType)):
            idxs.append(i)
        elif isinstance(ty, torch.ListType):
            if isinstance(ty.getElementType(), (torch.TensorType, torch.NumberType)):
                idxs.append(i)
    
    return idxs

def get_encoding_scale(n):
    if not hasattr(n, "meta"):
        return None # Not encoded
    if "encoding_scale" not in n.meta:
        return -1 # Not determined yet
    return n.meta["encoding_scale"]

def replace_node_helper(old, new, blacklist=[]):
    # We are using an helper because replace_all_uses_with
    # has a bug, which replaces the occurrence of old in new as well.
    # This is not the implied behavior in their tutorial.
    # We can further extend nodes to not update with blacklist.
    bl = blacklist + [new]
    #bl.append(new)

    old.replace_all_uses_with(new)

    for node in bl:
        # For nodes in blacklist, put back old
        idxs = []
        for i, n in enumerate(node.args):
            if n == new:
                idxs.append(i)
        new_args = list(node.args)
        for i in idxs:
            new_args[i] = old
        node.args = tuple(new_args)


def split_bias(mod):
    # ConvBias -> Conv + Bias, LinearBias -> Linear + Bias
    # I first made these into passes,
    # but PassManager seems to have an issue with a dynamic shape.
    # Instead of fixing it, I decided to just write it in a more robust way.
    replacements = []
    for node in mod.graph.nodes:
        if node.op == "call_function":
            prefix, op_name, mode = get_op_info(node.target)
            if op_name in ["conv2d", "linear"]:
                if node.args[2] is not None:
                    new_args = list(node.args)
                    bias = new_args[2]
                    new_args[2] = None
                    node.args = tuple(new_args)
                    with mod.graph.inserting_before(node):
                        bias_owner = bias.meta["owner"]
                        bias = mod.graph.call_function(torch.ops.aten.unsqueeze.default, args=(bias, 0))
                        bias.meta["owner"] = bias_owner
                        if op_name == "conv2d":
                            bias = mod.graph.call_function(torch.ops.aten.unsqueeze.default, args=(bias, 2))
                            bias.meta["owner"] = bias_owner
                            bias = mod.graph.call_function(torch.ops.aten.unsqueeze.default, args=(bias, 3))
                            bias.meta["owner"] = bias_owner

                    with mod.graph.inserting_after(node):
                        add = mod.graph.call_function(torch.ops.aten.add.Tensor, args=(node, bias))
                        copy_passes_meta(node, add)
                    replacements.append((node, add))
                    add.meta["owner"] = node.meta["owner"]

    for old, new in replacements:
        replace_node_helper(old, new)

    mod.graph.eliminate_dead_code()
    mod.graph.lint()
    mod.recompile()
    
def remove_asserts(mod):
    asserts = []
    for node in mod.graph.nodes:
        if node.op == "call_function":
            prefix, op_name, mode = get_op_info(node.target)
            
            if op_name == "_assert_tensor_metadata":
                # print(node)
                asserts.append(node)

    for node in asserts:
        mod.graph.erase_node(node)

    mod.graph.eliminate_dead_code()
    mod.graph.lint()
    mod.recompile()

def lower(mod, rank=0):
    # This is a bit hacky, but this stores the mapping between the original and the encrypted/encoded weights.
    # This is only useful for training.
    param_map = {}
    
    remove_asserts(mod)

	# Remove assert_tensor_metadata, they are newly introduced thing that do not like my compiler
    node_to_remove = []
    for node in mod.graph.nodes:
        if node.op == "call_function":
            prefix, op_name, mode = get_op_info(node.target)
            if op_name == "_assert_tensor_metadata":
                node_to_remove.append(node)
    for n in node_to_remove:
        mod.graph.erase_node(n)

    # First, split ConvBias and LinearBias
    split_bias(mod)

    # Replace < 0 to ltz (This is done at the beginning to not having to deal with this as a binary op later).
    for node in mod.graph.nodes:
        if node.op == "call_function":
            prefix, op_name, mode = get_op_info(node.target)
            if op_name == "lt" and node.args[1] == 0:
                node.target = torch.ops.cryptorch.ltz
                node.args = (node.args[0],)

                if "max_abs" in node.meta:
                    node.kwargs = {"max_abs": node.meta["max_abs"]}

    # Mark all the leaf nodes as encoding_scale = None
    
    # I am making this simpler, by always only allowing inputs/weights to
    # be a secret, and always encoding or encrypting all the inputs/weights.
    replacements = []
    for node in mod.graph.nodes:
        if node.op in ["placeholder", "get_attr"]:
            node.meta["encoding_scale"] = None
            if is_secret(node):
            #if "is_secret" in node.meta and node.meta["is_secret"]:
                src = get_src(node)
                assert(src is not None)
                # print(f"Encrypting {node=}, {src=}")
                if node.op == "placeholder":
                    with mod.graph.inserting_after(node):
                        encrypted = mod.graph.call_function(torch.ops.cryptorch.encrypt, args=(node, log_encoding_scale(), src))
                        encrypted.meta["encoding_scale"] = encoding_scale()
                        encrypted.meta["owner"] = node.meta["owner"]
                    replacements.append((node, encrypted))
                else:
                    orig_tensor = fetch_attr(node.target, mod)
                    new_tensor = torch.ops.cryptorch.encrypt(orig_tensor, src=src, precision=log_encoding_scale())
                    #setattr(mod, f"{node}_encrypted", new_tensor)
                    mod.register_parameter(f"{node}_encrypted", torch.nn.Parameter(new_tensor, requires_grad=False))
                    param_map[node.target] = f"{node}_encrypted"
                    node.target = f"{node}_encrypted"
                    node.meta["encoding_scale"] = encoding_scale()
            else:
                # print(f"Encoding {node=}")
                if node.op == "placeholder":
                    with mod.graph.inserting_after(node):
                        encoded = mod.graph.call_function(torch.ops.cryptorch.encode, args=(node, encoding_scale()))
                        encoded.meta["encoding_scale"] = encoding_scale()
                        encoded.meta["owner"] = node.meta["owner"]
                    replacements.append((node, encoded))
                else:
                    orig_tensor = fetch_attr(node.target, mod)
                    new_tensor = getattr(orig_tensor * encoding_scale(), ring_dtype())()
                    #setattr(mod, f"{node}_encoded", new_tensor)
                    mod.register_parameter(f"{node}_encoded", torch.nn.Parameter(new_tensor, requires_grad=False))
                    param_map[node.target] = f"{node}_encoded"
                    node.target = f"{node}_encoded"
                    node.meta["encoding_scale"] = encoding_scale()

    for node in mod.graph.nodes:
        if node.op == "call_function":
            prefix, op_name, mode = get_op_info(node.target)
            if prefix == "cryptorch" and op_name == "ltz":
                # LTZ always output scale 1
                node.meta["encoding_scale"] = 1

    # Encode secrets.
    #for node in mod.graph.nodes:
    #    #if node.name in secret_tensors:
    #    if "is_secret" in node.meta and node.meta["is_secret"]:
    #        src = get_src(node)
    #        assert(src is not None)
    #        print(f"Encrypting {node=}, {src=}")
    #        with mod.graph.inserting_after(node):
    #            encoded = mod.graph.call_function(torch.ops.cryptorch.encrypt, args=(node, log_encoding_scale, src))
    #            encoded.meta["encoding_scale"] = encoding_scale
    #            encoded.meta["owner"] = node.meta["owner"]
    #        replacements.append((node, encoded))

    for old, new in replacements:
        replace_node_helper(old, new)

    mod.graph.eliminate_dead_code()
    mod.graph.lint()
    mod.recompile()

    # Do another forward pass and propagate the encoding information. 
    # If a node takes multiple inputs with different encodings, it follows the rule from CrypTen.
    # 1. If the op is between encoded data and non-encoded data
    # 1.1. If the scale is 1 and the other is float:
    # 1.1.1. If it is additive, encode both with the default scale.
    # 1.1.2. Otherwise, it is sufficient to encode the float with the default scale (following div not needed).
    # 1.2. If the scale is 1 and the other is int:
    # 1.2.1. Don't do anything, and keep the scale to 1 (following div not needed).
    # 1.3. If the scale is > 1 and the other is float:
    # 1.3.1. Encode the float with the scale (If non-additive, div needed)
    # 1.4. If the scale is > 1 and the other is int:
    # 1.4.1. If it is additive, encode the int with the scale.
    # 1.4.2. Otherwise, simply set the scale out the output (no div needed).
    # 1.5. Then, perform the op.

    # 2. If the op is between two encoded vals
    # 2.1. If additive, make the scale same by following the larger scale.
    # 2.2. If non-additive, just do the op

    # 3. For non-additive func, we need to do postprocessing to reduce the scale.
    # 3.1. If it was between encoded and non-encoded val (case 1), look at above to see when div is needed.
    # 3.2. If this was between two encoded vals,
    # 3.2.1. If both's scales are > 1, use the rule from CrypTen.
    # 3.2.2. If only one's scale is > 1, simply use that as the scale moving forward. No need to divide.
    # We don't do while loop assuming the graph traversal is done in the "right" order.
    changed = True
    # print(mod)
    while changed:
        changed = False
        replacements = []
        for node in mod.graph.nodes:
            #print(f"{node=}")
            if "encoding_scale" in node.meta:
                # This node is already processed.
                # TODO: Can the scale possible change after multiple iter? I don't think so.
                continue
            elif node.op == "call_function":
                prefix, op_name, mode = get_op_info(node.target)
                #print(prefix, op_name, mode)
                if prefix == "cryptorch" and (op_name in ["encode", "ltz", "encrypt"]):
                    # I think this is not necessary because if encoding scale is set, there's a continue above.
                    assert(False)
                    continue
                scales = []
                skip = False
                # Ignore params like stride, padding, etc.
                idxs = get_compute_input_idxs(node.target, len(node.args))

                if len(idxs) == 1:
                    # This is either unary op or ops where the input is a list (e.g., sum, stack, etc.)
                    # TODO
                    arg = node.args[idxs[0]]
                    if isinstance(arg, list):
                        scales = []
                        for n in arg:
                            scales.append(get_encoding_scale(n))
                        # If encoding scales are different inside the list,
                        # we must make them the same. Currently, leaving as future TODO.
                        s = set(scales)
                        assert(len(s) == 1)
                        node.meta["encoding_scale"] = scales[0]
                    else:
                        node.meta["encoding_scale"] = get_encoding_scale(arg)
                elif len(idxs) == 2:
                    for i in idxs:
                        n = node.args[i]
                        scale = get_encoding_scale(n)
                        #print(f"{i=} {n=} {type(n)=} {scale=}")
                        if scale == -1:
                            # Parent node needs to be processed first
                            skip = True
                            break
                        scales.append(scale)
                    if skip:
                        continue

                    # Check if all the input encodings are the same
                    s = set(scales)
                    if len(s) == 1:
                        node.meta["encoding_scale"] = scales[0]
                        if len(idxs) == 2 and not is_additive(op_name) and scales[0] is not None:
                            # Non-additive binary op between two args.
                            # We need to do postprocessing to reduce the scale.
                            with mod.graph.inserting_after(node):
                                div = mod.graph.call_function(
                                    torch.ops.cryptorch.div, 
                                    args=(node, node.meta["encoding_scale"])
                                )
                                div.meta["owner"] = node.meta["owner"]
                                copy_passes_meta(node, div)
                                replacements.append((node, div))
                            div.meta["encoding_scale"] = node.meta["encoding_scale"]
                    elif op_name in ["div", "div_"]:
                        # Div is a special case.
                        # The second argument cannot be a secret at this point.
                        # TODO: Div value should be int or near-int.
                        # Otherwise, it should have been replaced with reciprocal mult.
                        #assert(isinstance(node.args[1], int))
                        new_args = list(node.args)
                        new_args[1] = int(node.args[1])
                        assert(new_args[1] == node.args[1])
                        node.args = tuple(new_args)
                        node.meta["encoding_scale"] = node.args[0].meta["encoding_scale"]
                        node.target = torch.ops.cryptorch.div
                    elif op_name in ["full_like"]:
                        # TODO: Handle other nodes like zeros-like, etc.
                        new_args = list(node.args)
                        new_args[1] *= encoding_scale()
                        node.args = tuple(new_args)
                        node.meta["encoding_scale"] = encoding_scale()
                        node.meta["owner"] = [True, True]
                    else:
                        # Handle the mismatch!
                        # After properly handling bias in the prev stage, this is at most binary op
                        if len(scales) != 2:
                            raise AssertionError(f"{scales=} for {node=} is more than length 2. What exactly is happening?")
                        # print(f"Handle mismatch {scales=}, {node=}")
                        div_needed = False
                        if scales[0] is None or scales[1] is None:
                            # 1. If the op is between encoded data and non-encoded data
                            if scales[0] is None:
                                enc_arg_idx = 1
                                other_arg_idx = 0
                            else:
                                enc_arg_idx = 0
                                other_arg_idx = 1
                            if scales[enc_arg_idx] == 1:
                                other = node.args[other_arg_idx]

                                # Handle scalar_tensor. Due to the lack of documentation, not exactly sure what this function does.
                                # I just delete it and replace it with float/int.
                                if isinstance(other, torch.fx.node.Node):
                                    if other.op == "call_function":
                                        prefix, op_name, mode = get_op_info(other.target)
                                        if op_name == "scalar_tensor":
                                            if other.kwargs["dtype"] == torch.float32:
                                                other = float(other.args[0])
                                            elif other.kwargs["dtype"] == torch.int64:
                                                other = int(other.args[0])
                                            else:
                                                raise AssertionError(other, other.kwargs)
                                        else:
                                            raise AssertionError(other, op_name)
                                    else:
                                        raise AssertionError(other, other.op)
                                elif isinstance(other, (float, int)):
                                    pass
                                else:
                                    raise AssertionError(other, type(other))

                                if isinstance(other, float):
                                    # 1.1. If the scale is 1 and the other is float:
                                    if is_additive(op_name):
                                        # 1.1.1. If it is additive, encode both with the default scale.
                                        with mod.graph.inserting_before(node):
                                            encoded = mod.graph.call_function(torch.ops.cryptorch.encode, args=(node.args[enc_arg_idx], encoding_scale()))
                                            encoded.meta["encoding_scale"] = encoding_scale()
                                            encoded.meta["owner"] = node.args[enc_arg_idx].meta["owner"]
                                            encoded2 = round(other * encoding_scale())
                                            new_args = list(node.args)
                                            new_args[enc_arg_idx] = encoded
                                            new_args[other_arg_idx] = encoded2
                                            node.args = tuple(new_args)
                                    else:
                                        # 1.1.2. Otherwise, it is sufficient to encode the float with the default scale (following div not needed).
                                        encoded2 = round(other * encoding_scale())
                                        new_args = list(node.args)
                                        new_args[other_arg_idx] = encoded2
                                        node.args = tuple(new_args)
                                    node.meta["encoding_scale"] = encoding_scale()
                                elif isinstance(other, int):
                                    # 1.2. If the scale is 1 and the other is int:
                                    # 1.2.1. Don't do anything, and keep the scale to 1 (following div not needed).
                                    node.meta["encoding_scale"] = 1
                                else:
                                    # TODO: Similary, must handle FloatTensor, IntTensor inputs.
                                    raise NotImplementedError(f"Handling of {node.args[other_arg_idx]} (type {type(node.args[other_arg_idx])}) not implemented!")
                            else:
                                if isinstance(node.args[other_arg_idx], float):
                                    # 1.3. If the scale is > 1 and the other is float:
                                    # 1.3.1. Encode the float with the scale (If non-additive, div needed)
                                    with mod.graph.inserting_before(node):
                                        encoded = round(node.args[other_arg_idx] * scales[enc_arg_idx])
                                        new_args = list(node.args)
                                        new_args[other_arg_idx] = encoded
                                        node.args = tuple(new_args)
                                        node.meta["encoding_scale"] = scales[enc_arg_idx]
                                        if not is_additive(op_name):
                                            div_needed = True
                                elif isinstance(node.args[other_arg_idx], int):
                                    # 1.4. If the scale is > 1 and the other is int:
                                    if is_additive(op_name):
                                        # 1.4.1. If it is additive, encode the int with the scale.
                                        encoded = node.args[other_arg_idx] * scales[enc_arg_idx]
                                        new_args = list(node.args)
                                        new_args[other_arg_idx] = encoded
                                        node.args = tuple(new_args)
                                        node.meta["encoding_scale"] = scales[enc_arg_idx]
                                    else:
                                        # 1.4.2. Otherwise, simply set the scale out the output (no div needed).
                                        node.meta["encoding_scale"] = scales[enc_arg_idx]
                                elif isinstance(node.args[other_arg_idx], torch.fx.node.Node):
                                    # When FloatTensor, Same as 1.3.
                                    # Note that when it is an IntTensor, we can do the same with 1.4,
                                    # Which is slightly more efficient (no trailing div), but we are not
                                    # doing that because this is the common case.
                                    with mod.graph.inserting_before(node):
                                        encoded = mod.graph.call_function(torch.ops.cryptorch.encode, args=(node.args[other_arg_idx], scales[enc_arg_idx]))
                                        encoded.meta["encoding_scale"] = encoding_scale()
                                        encoded.meta["owner"] = node.args[other_arg_idx].meta["owner"]
                                        new_args = list(node.args)
                                        new_args[other_arg_idx] = encoded
                                        node.args = tuple(new_args)
                                        node.meta["encoding_scale"] = scales[enc_arg_idx]
                                        if not is_additive(op_name):
                                            div_needed = True
                                    '''
                                    elif isinstance(node.args[other_arg_idx], torch.IntTensor):
                                        # Same as 1.4.
                                        if is_additive(op_name):
                                            # 1.4.1. If it is additive, encode the int with the scale.
                                            encoded = mod.graph.call_function(torch.ops.cryptorch.encode, args=(node.args[other_arg_idx], scales[enc_arg_idx]))
                                            encoded.meta["encoding_scale"] = encoding_scale
                                            encoded.meta["owner"] = node.args[other_arg_idx].meta["owner"]
                                            new_args = list(node.args)
                                            new_args[other_arg_idx] = encoded
                                            node.args = tuple(new_args)
                                            node.meta["encoding_scale"] = scales[enc_arg_idx]
                                        else:
                                            # 1.4.2. Otherwise, simply set the scale out the output (no div needed).
                                            node.meta["encoding_scale"] = scales[enc_arg_idx]
                                    '''
                                else:
                                    raise NotImplementedError()
                        else:
                            # 2. Both args are encoded
                            if is_additive(op_name):
                                # 2.1. If additive, make the scale same by following the larger scale.
                                # print(f"{op_name=} Additive")
                                if scales[0] == scales[1]:
                                    node.meta["encoding_scale"] = scales[0]
                                else:
                                    if scales[0] > scales[1]:
                                        l, s = 0, 1
                                    else:
                                        l, s = 1, 0
                                    # 1.2. Encode the arg with the smaller scale with the larger scale
                                    #with mod.graph.inserting_after(node.args[s]):
                                    with mod.graph.inserting_before(node):
                                        assert(scales[l] % scales[s] == 0)
                                        encoded = mod.graph.call_function(torch.ops.cryptorch.encode, args=(node.args[s], scales[l] // scales[s]))
                                        encoded.meta["owner"] = node.args[s].meta["owner"]
                                        encoded.meta["encoding_scale"] = scales[l]
                                    #replacements.append((node.args[1], encoded))
                                    new_args = list(node.args)
                                    new_args[s] = encoded
                                    node.args = tuple(new_args)
                                    node.meta["encoding_scale"] = scales[l]
                            else:
                                # print(f"{op_name=} NOT Additive")
                                # 2.2. If non-additive, just do the op
                                pass
                        if not is_additive(op_name):
                            # 3. For non-additive func, we need to do postprocessing to reduce the scale.
                            if scales[0] is None or scales[1] is None:
                                # 3.1. If it was between encoded and non-encoded val (case 1), look at above to see when div is needed.
                                if div_needed:
                                    with mod.graph.inserting_after(node):
                                        div = mod.graph.call_function(
                                            torch.ops.cryptorch.div,
                                            args=(node, node.meta["encoding_scale"])
                                        )
                                        div.meta["owner"] = node.meta["owner"]
                                        copy_passes_meta(node, div)
                                        replacements.append((node, div))
                                    div.meta["encoding_scale"] = node.meta["encoding_scale"]
                            else:
                                # 3.2. If this was between two encoded vals,
                                if node.args[0].meta["encoding_scale"] > 1 and node.args[1].meta["encoding_scale"] > 1:
                                    # 3.2.1. If both's scales are > 1, divide with TODO.
                                    assert(node.args[0].meta["encoding_scale"] == node.args[1].meta["encoding_scale"])
                                    with mod.graph.inserting_after(node):
                                        div = mod.graph.call_function(
                                            torch.ops.cryptorch.div,
                                            args=(node, node.meta["encoding_scale"])
                                        )
                                        div.meta["owner"] = node.meta["owner"]
                                        copy_passes_meta(node, div)
                                        replacements.append((node, div))
                                    div.meta["encoding_scale"] = node.meta["encoding_scale"]
                                elif node.args[0].meta["encoding_scale"] > 1:
                                    # 3.2.2. If only one's scale is > 1, simply use that as the scale moving forward. No need to divide.
                                    if "encoding_scale" in node.meta:
                                        assert(node.args[0].meta["encoding_scale"] == node.meta["encoding_scale"])
                                    else:
                                        node.meta["encoding_scale"] = node.args[0].meta["encoding_scale"]
                                elif node.args[1].meta["encoding_scale"] > 1:
                                    # 3.2.2. If only one's scale is > 1, simply use that as the scale moving forward. No need to divide.
                                    if "encoding_scale" in node.meta:
                                        assert(node.args[1].meta["encoding_scale"] == node.meta["encoding_scale"])
                                    else:
                                        node.meta["encoding_scale"] = node.args[1].meta["encoding_scale"]
                                else:
                                    raise NotImplementedError()

                else:
                    raise AssertionError(f"Data input idxs should be <=2, but currently {len(idxs)} for {node}. What is happening?")

                # print(f"{op_name=}, {len(idxs)=}, {is_secret(node.args[0])=}")
                # print(f"{node.args[0]}")
                # Replace ops to MPC-specific ops if needed.
                if len(idxs) == 2 and not is_additive(op_name):
                    # Non-additive binary op between two args.
                    if sum([is_secret(node.args[i]) for i in idxs]) == 2:
                        # print(f"{op_name} needs to be using MPC-specific functions!")
                        if op_name == "mul":
                            # Replace mul with square if possible, because that is faster.
                            if node.args[0] == node.args[1]:
                                node.target = torch.ops.cryptorch.square
                                node.args = tuple([node.args[0]])
                            else:
                                node.target = torch.ops.cryptorch.mul
                        elif op_name == "mul_":
                            if node.args[0] == node.args[1]:
                                node.target = torch.ops.cryptorch.square_
                                node.args = tuple([node.args[0]])
                            else:
                                node.target = torch.ops.cryptorch.mul_
                        elif op_name == "conv2d":
                            node.target = torch.ops.cryptorch.conv2d
                        elif op_name == "linear":
                            node.target = torch.ops.cryptorch.linear
                        elif op_name == "matmul":
                            node.target = torch.ops.cryptorch.matmul
                        elif op_name == "mm":
                            #print(f"{node} being handled!")
                            node.target = torch.ops.cryptorch.matmul
                        else:
                            raise NotImplementedError(op_name)
                elif len(idxs) == 1 and is_secret(node.args[0]):
                    # Non-binary ops that still needs MPC.
                    if op_name == "amax":
                        node.target = torch.ops.cryptorch.amax
                        if "max_abs" in node.meta:
                            node.kwargs = {"max_abs": node.meta["max_abs"]}
                    elif op_name == "max_pool2d":
                        # print("HIT")
                        node.target = torch.ops.cryptorch.max_pool2d
                        if "max_abs" in node.meta:
                            node.kwargs = {"max_abs": node.meta["max_abs"]}
                    elif op_name == "adaptive_avg_pool2d":
                        node.target = torch.ops.cryptorch.adaptive_avg_pool2d
            elif node.op == "output":
                # TODO
                pass
            else:
                raise NotImplementedError(node.op)

        for old, new in replacements:
            replace_node_helper(old, new)

    mod.graph.eliminate_dead_code()
    mod.graph.lint()
    mod.recompile()

    # MPC-related transformations.
    # Properly replace binary ops, dropout, and mean.
    # Patch scalar addition for party 1.
    replacements = []
    for node in mod.graph.nodes:
        if node.op == "call_function":
            prefix, op_name, mode = get_op_info(node.target)
            # Remove conversions (MPC backend always work with integer ring)
            # and dropout.
            if op_name in ["to", "dropout"]:
                replacements.append((node, node.args[0], []))
            elif op_name in ["add", "add_", "sub", "sub_", "rsub"]:
                # If y is not a secret
                if not is_secret(node.args[1]):
                    # Only rank 0 adds or subs the real y.
                    if rank != 0:
                        new_args = list(node.args)
                        new_args[1] = 0
                        node.args = tuple(new_args)
            elif op_name == "bitwise_not":
                new_args = list(node.args)
                assert(len(new_args) == 1)
                if rank != 0:
                    new_args.append(0)
                else:
                    new_args.append(1)
                node.args = tuple(new_args)
                node.target = torch.ops.aten.rsub.Scalar
            elif op_name == "mean":
                if mode == "dim":
                    node.target = torch.ops.aten.sum.dim_IntList
                    with mod.graph.inserting_after(node):
                        # Shape infos are public values
                        numel1 = mod.graph.call_function(torch.ops.aten.numel, args=(node.args[0],))
                        numel1.meta["owner"] = (True, True)
                        numel1.meta["encoding_scale"] = 1
                        copy_passes_meta(node, numel1)

                        numel2 = mod.graph.call_function(torch.ops.aten.numel, args=(node,))
                        numel2.meta["owner"] = (True, True)
                        numel2.meta["encoding_scale"] = 1
                        copy_passes_meta(node, numel2)

                    with mod.graph.inserting_after(numel1):
                        # This is just intdiv, not MPC div
                        divisor = mod.graph.call_function(torch.ops.aten.div.Tensor_mode, args=(numel1, numel2), kwargs={"rounding_mode": "trunc"})
                        divisor.meta["owner"] = (True, True)
                        divisor.meta["encoding_scale"] = 1
                        copy_passes_meta(node, divisor)

                    with mod.graph.inserting_after(divisor):
                        div = mod.graph.call_function(
                            torch.ops.cryptorch.div,
                            args=(node, divisor),
                        )
                        div.meta["owner"] = node.meta["owner"]
                        div.meta["encoding_scale"] = node.meta["encoding_scale"]
                        copy_passes_meta(node, div)
                        # If I do replacement here, numel2's arg is also replaced. So, I do this manually.
                        replacements.append((node, div, [numel2]))
                elif mode == "default":
                    node.target = torch.ops.aten.sum.default
                    with mod.graph.inserting_after(node):
                        # Shape infos are public values
                        numel1 = mod.graph.call_function(torch.ops.aten.numel, args=(node.args[0],))
                        numel1.meta["owner"] = (True, True)
                        numel1.meta["encoding_scale"] = 1
                        copy_passes_meta(node, numel1)

                    with mod.graph.inserting_after(numel1):
                        div = mod.graph.call_function(
                            torch.ops.cryptorch.div, 
                            args=(node, numel1),
                            kwargs={"mode": truncation_mode}
                        )
                        div.meta["owner"] = node.meta["owner"]
                        div.meta["encoding_scale"] = node.meta["encoding_scale"]
                        copy_passes_meta(node, div)
                        # If I do replacement here, numel2's arg is also replaced. So, I do this manually.
                        replacements.append((node, div, []))
                else:
                    raise NotImplementedError(prefix, op_name, mode)
        elif node.op == "output":
            with mod.graph.inserting_before(node):
                precisions = []
                for n in node.args[0]:
                    if "encoding_scale" in n.meta:
                        precision = (n.meta["encoding_scale"] - 1).bit_length()
                    else:
                        precision = -1
                    precisions.append(precision)
                dec = mod.graph.call_function(torch.ops.cryptorch.decrypt_sequence, args=(node.args[0], precisions, list(node.meta["output_owners"])))
                dec.meta["owner"] = node.meta["owner"]
                dec.meta["encoding_scale"] = None
            node.args = (dec,)
            node.meta["encoding_scale"] = None
    for old, new, bl in replacements:
        replace_node_helper(old, new, bl)

    '''
    for node in mod.graph.nodes:
        if node.op == "output":
            with mod.graph.inserting_before(node):
                dec = mod.graph.call_function(torch.ops.cryptorch.decrypt, args=(node.args[0][0], log_encoding_scale))
                dec.meta["owner"] = node.meta["owner"]
                dec.meta["encoding_scale"] = None
            node.args = ((dec,),)
            node.meta["encoding_scale"] = None
    '''
    
    mod.graph.eliminate_dead_code()
    mod.graph.lint()
    mod.recompile()
    
    #return mod
    return param_map
