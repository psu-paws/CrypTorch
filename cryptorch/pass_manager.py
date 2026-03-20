from torch.fx._symbolic_trace import symbolic_trace
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from executorch.exir.pass_base import ExportPass
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor
import types
from torch.fx.passes.shape_prop import ShapeProp
from cryptorch.passes import TunablePass
import copy
from cryptorch.knob_tuner import BinarySearchGreedyKnobTuner, KnobTuner, LinearGreedyKnobTuner, HillClimbingKnobTuner
from cryptorch.utils import get_head, _export_module, get_inputs
import itertools
from torch.fx.traceback import NodeSource, NodeSourceAction

# This is a minor tweak of torch.fx.subgraph_rewriter
# to meet my need.
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    TYPE_CHECKING,
    Union,
    Iterable
)
from torch.fx import Node
from torch.fx.subgraph_rewriter import _replace_attributes

def replace_matches(gm, match, replacement, inputs, pass_name=None):
    original_graph = gm.graph

    match_changed_node: Dict[Node, Node] = {}
    if isinstance(replacement, nn.Module):
        replacement_graph = _export_module(replacement, inputs).graph
    elif isinstance(replacement, torch.fx.graph_module.GraphModule):
        replacement_graph = replacement.graph
    elif isinstance(replacement, types.FunctionType):
        replacement_graph = symbolic_trace(replacement).graph

    replacement_placeholders = [
        n for n in replacement_graph.nodes if n.op == "placeholder"
    ]

    # Build connecting between replacement graph's input and original graph input producer node

    # Initialize `val_map` with mappings from placeholder nodes in
    # `replacement` to their corresponding node in `original_graph`
    # TODO: This part breaks when an immutable is used as an input. Adding a rather hacky solution for now.
    match_placeholder_nodes = []
    for n in match.placeholder_nodes:
        if isinstance(n, (torch.fx.immutable_collections.immutable_list, list)):
            match_placeholder_nodes += list(n)
        else:
            match_placeholder_nodes.append(n)
    #print(match_placeholder_nodes, replacement_placeholders)

    if pass_name is not None:
        # print(match.nodes_map[match.anchors[0]].meta)
        pass_labels = list(itertools.chain.from_iterable(map(lambda n: match.nodes_map[n].meta["passes"] if "passes" in match.nodes_map[n].meta else [], match.anchors)))
        pass_labels.append(pass_name)
        # print(f"AAAA: {pass_labels}")
    
    node_sources = []
    for n in match.anchors:
        old_node = match.nodes_map[n]
        node_sources.append(NodeSource(
            node=old_node,
            pass_name=f"cryptorch.{pass_name}" if pass_name is not None else "cryptorch.unknown_pass",
            action=NodeSourceAction.REPLACE
        ))

    if not len(match_placeholder_nodes) == len(replacement_placeholders):
        assert(len(match_placeholder_nodes) > len(replacement_placeholders))
        print(f"Copying the last {len(match_placeholder_nodes) - len(replacement_placeholders)} nodes in {match_placeholder_nodes=} to {replacement_placeholders=}. Is this what you want?")
        replacement_placeholders += match_placeholder_nodes[:len(replacement_placeholders) - len(match_placeholder_nodes)]
    val_map: Dict[Node, Node] = {}
    for rn, gn in zip(replacement_placeholders, match_placeholder_nodes):
        if isinstance(gn, Node):
            val_map[rn] = match_changed_node.get(gn, gn)
            if gn != val_map[rn]:
                # Update match.placeholder_nodes and match.nodes_map with the node that replaced gn
                gn_ind = match_placeholder_nodes.index(gn)
                match_placeholder_nodes[gn_ind] = match_changed_node[gn]
                map_key = list(match.nodes_map.keys())[
                    list(match.nodes_map.values()).index(gn)
                ]
                match.nodes_map[map_key] = match_changed_node[gn]
        else:
            val_map[rn] = gn
    
    # mark all nodes with pass
    if pass_name is not None:
        for node in replacement_graph.nodes:
            node.meta["passes"] = pass_labels
            # print(node.meta["passes"])
    
    for node in replacement_graph.nodes:
            node.meta["from_node"] = copy.deepcopy(node_sources)
    
    if pass_name is not None:
        for node in val_map.values():
            if isinstance(node, Node):
                if "input_of" not in node.meta:
                    node.meta["input_of"] = list()
                node.meta["input_of"].append(pass_name)

    # Copy the replacement graph over
    user_nodes: Set[Node] = set()
    for n in match.returning_nodes:
        user_nodes.update(n.users)
    assert user_nodes, "The returning_nodes should have at least one user node"

    if len(user_nodes) == 1:
        first_user_node = next(iter(user_nodes))
    else:
        # If there are multiple user nodes, we need to find the first user node
        # in the current execution order of the `original_graph`
        for n in original_graph.nodes:
            if n in user_nodes:
                first_user_node = n
                break

    with original_graph.inserting_before(first_user_node):  # type: ignore[possibly-undefined]
        copied_returning_nodes = original_graph.graph_copy(
            replacement_graph, val_map
        )

    if isinstance(copied_returning_nodes, Node):
        copied_returning_nodes = (copied_returning_nodes,)
        
    if pass_name is not None:
        for node in copied_returning_nodes:
            if isinstance(node, Node):
                if "output_of" not in node.meta:
                    node.meta["output_of"] = list()
                node.meta["output_of"].append(pass_name)
        

    # Get a list of nodes that have been replaced into the graph
    replacement_nodes: List[Node] = [
        v for v in val_map.values() if v not in match.placeholder_nodes
    ]

    # Hook the output Node of the replacement subgraph in to the
    # original Graph at the correct location
    assert len(match.returning_nodes) == len(copied_returning_nodes)  # type: ignore[arg-type]
    for gn, copied_node in zip(match.returning_nodes, copied_returning_nodes):  # type: ignore[arg-type]
        gn.replace_all_uses_with(copied_node)
        match_changed_node[gn] = copied_node

    # Remove the original nodes
    #for node in reversed(pattern_graph.nodes):
    for node in match.nodes_map:
        if node.op != "placeholder" and node.op != "output":
            gn = match.nodes_map[node]
            gm.graph.erase_node(gn)

    # Update the passed-in GraphModule to reflect the new state of
    # `original_graph`
    gm.recompile()

    # If `replacement` was an nn.Module, we'll need to make sure that
    # all the submodules have been copied over correctly
    if isinstance(replacement, torch.nn.Module):
        _replace_attributes(gm, replacement)

    return replacement_nodes

class TestInterpreter(torch.fx.Interpreter):
    #def __init__(self, mod, encrypt_output_list=[], plain_weight_list="all", rank=0):
    def __init__(self, *args, **kwags):
        super().__init__(*args, **kwags)

    def run_node(self, n : Node) -> Any:
        # Get any inputs being passed through a higher-level module
        #print(f"Run {n} {n.op} {n.target} {n.meta['owner']}")
        print(f"Run {n} {n.op} {n.target}")
        inputs, kwargs = super().fetch_args_kwargs_from_env(n)
        #print("Inputs", [[(type(y), y.shape) for y in x] if isinstance(x, tuple) else ((type(x), x.shape) if hasattr(x, "shape") else (type(x), x)) for x in inputs])
        print("Inputs", [[(type(y), y.shape) for y in x] if isinstance(x, tuple) else ((type(x), x.shape) if hasattr(x, "shape") else (type(x), x)) for x in inputs])
        return super().run_node(n)

    def run(self, *args):
        return super().run(*args)

class NVTXInterpreter(torch.fx.Interpreter):
    #def __init__(self, mod, encrypt_output_list=[], plain_weight_list="all", rank=0):
    def __init__(self, *args, **kwags):
        super().__init__(*args, **kwags)

    def run_node(self, n : Node) -> Any:
        with torch.cuda.nvtx.range(f"OP-{n.name}-{n.meta['passes'] if 'passes' in n.meta else []}"):
            result = super().run_node(n)
        return result

    def run(self, *args):
        return super().run(*args)


class TunerConfig:
    def __init__(
        self,
        tuner: Union[str, KnobTuner],
        search_inputs,
        input_preprocess_func = lambda x: x,
        objective_func = torch.nn.CrossEntropyLoss(),
        objective_threshold = 0.05
    ):
        if isinstance(tuner, str):
            if tuner.casefold() == "greedy".casefold():
                self.tuner = LinearGreedyKnobTuner()
            elif tuner.casefold() == "hillclimbing".casefold():
                self.tuner = HillClimbingKnobTuner()
            elif tuner.casefold() == "greedy_binary".casefold():
                self.tuner = BinarySearchGreedyKnobTuner()
            else:
                raise ValueError(f"Unknown tuner: {tuner}")
        else:
            self.tuner = tuner
        
        self.search_inputs = search_inputs
        self.input_preprocess_function = input_preprocess_func
        self.objective_func = objective_func
        self.objective_threshold = objective_threshold

class CrypTorchPassManager:
    def __init__(
        self, 
        module,
        pass_list, 
        num_parties=2, 
        tuner_config: TunerConfig = None):
        self.module = module
        self.pass_list = pass_list
        self.num_parties = num_parties
        self.tuner_config = tuner_config

    def eval_module(self):
        import tqdm
        if self.tuner_config is None:
            raise RuntimeError("eval_module called when self.tuner_config is None")
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(self.tuner_config.search_inputs)):
                x, y_tmp = self.tuner_config.input_preprocess_function(batch)
                # y_tmp = self.get_y(batch)
                pred_tmp = self.module(*x)
                if i == 0:
                    pred = pred_tmp
                    y = y_tmp
                else:
                    pred = torch.cat([pred, pred_tmp])
                    y = torch.cat([y, y_tmp])
        return self.tuner_config.objective_func(pred, y)

    def run(self, set_knobs_manually=[], *, no_tune=False, return_best_knobs=False, verbose=True):
        
        if verbose:
            verbose_print = print
        else:
            def verbose_print(*args, **kwargs):
                pass
        # Hacks to track the shape. Not sure if this part is still needed with the new pass manager code,
        # but keeping just to be on the safe side.
        ep = ExportPass()
        
        # Function for recomputing all the ensor shapes.
        # Should be called after each pass that made modifications to the graph.
        def propagate_tensor_shapes():
            # This is a bit hacky, but use ExportPass's helper function to get the input of the graph module.
            inputs = tuple(ep.inputs(self.module))
            
            # This part of the code is from executorch/exir/pass_base.py
            fake_tensor_mode = None
            for i in inputs:
                if isinstance(i, FakeTensor):
                    assert (
                        fake_tensor_mode is None or fake_tensor_mode is i.fake_mode
                        ), "Multiple fake tensor mode detected."
                    fake_tensor_mode = i.fake_mode
            # Borrowed code end
            
            # Propergate Tensor Shapes ussing ShapeProp
            intp = ShapeProp(self.module, fake_mode=fake_tensor_mode)
            intp.garbage_collect_values = False
            intp.propagate(*inputs)
            
            return intp

        
        intp = propagate_tensor_shapes()
        
        # check if there are any tunable passes
        do_tuning = (self.tuner_config is not None) and (len(set_knobs_manually) == 0) and (any(map(lambda x: isinstance(x, TunablePass), self.pass_list))) and (not no_tune)
        verbose_print(f"{do_tuning=}")
        verbose_print(f"{(self.tuner_config is not None)=}")
        verbose_print(f"{(len(set_knobs_manually) == 0)=}")
        verbose_print(f"{(any(map(lambda x: isinstance(x, TunablePass), self.pass_list)))=}")
        verbose_print(f"{(not no_tune)=}")
        
        '''
        if changed:
            print(f"Pass {p!r} applied")
            
            # recompute tensor shapes now that changes have been made
            # not always running this massively speeds up the passes
        
            for p, count in pass_stat.items():
                print(f"Pass {p!r} applied a total of {count} locations")    
        return changed
        '''

        changed = True

        # First, run non-tunable passes.
        while changed:
            changed = False
            for p in list(filter(lambda p: not isinstance(p, TunablePass), self.pass_list)):
                _match = self.find_match_locations(p)
                if len(_match) == 0:
                    continue
                changed = True
                self.apply_passes([(p, m, get_inputs(m, intp)) for m in _match], [])
                intp = propagate_tensor_shapes()

        # Second, try global approximations of all the Tunable passes.
        # Note that this doesn't work correctly if the TunablePass calls
        # another TunablePass (e.g., ReciprocalPass may do this).
        
        if do_tuning:
            ref_acc = self.eval_module()
            verbose_print(f"Reference objective value: {ref_acc}")

        final_knobs = []
        
        changed = True
        while changed:
            changed = False

            # Apply all the tunable passes at once.
            best_mod = None
            best_acc = None
            best_state = None # This is only for printing
            
            if do_tuning:
                tuner = self.tuner_config.tuner
                tuner.reset()
            # Knob values for each pass location
            knobs = []
            i = 0
            while True:
                # Find all the matching locations of all the tunable passes.
                _match = []
                for p in list(filter(lambda p: isinstance(p, TunablePass), self.pass_list)):
                    _tmp_match = self.find_match_locations(p)
                    if len(_tmp_match) == 0:
                        continue
                    changed = True
                    # If the tuner internally calculates cost,
                    # it needs to know the input shapes, so save and
                    # pass the input shapes. Note that this must be
                    # calculated before we modify self.module.
                    # Initially, have knobs to their max values,
                    # and comparisons all turned on.
                    _match += [(p, m, get_inputs(m, intp)) for m in _tmp_match]
                    if len(knobs) < len(_match):
                        knobs += [p.get_max_knob()] * len(_tmp_match)
                if do_tuning and len(tuner.cur_state) == 0:
                    tuner.cur_state = knobs + []
                assert(len(knobs) == len(_match))
                if len(_match) == 0:
                    verbose_print(f"No matches to apply the pass")
                    break

                # Code for manual knob set
                if len(set_knobs_manually) > 0:
                    verbose_print(f"{len(set_knobs_manually)=}")
                    verbose_print(f"{len(knobs)=}")
                    assert(len(set_knobs_manually) == len(knobs))
                    knobs = set_knobs_manually

                verbose_print(f"Applying passes with knobs={knobs}")
                # 1. Back the current module
                mod_bak = copy.deepcopy(self.module)

                #print("Before applying pass")
                #print(self.module)

                # 1. Try applying the pass
                self.apply_passes(_match, knobs)
                
                if not do_tuning:
                    best_mod = self.module
                    best_state = knobs
                    break

                #print("After applying pass")
                #print(self.module)

                # 2. Evaluate
                objective = self.eval_module()

                print(f"Objective: {objective}")

                # If acc is still above a threshold, try more aggressive knob.
                if best_acc is None:
                    print(f"This is the best objective you can achieve with the best approximation")
                    best_acc = objective
                    last_attempt_successful = True
                #elif acc == best_acc:
                #    print(f"This is not worse than the best acc you can achieve, so this is strictly better.")
                #    last_attempt_successful = True
                #elif (acc > best_acc * self.acc_thres) if self.acc_thres < 1.0 else (acc < best_acc * self.acc_thres):
                elif abs(objective - best_acc) / best_acc <= self.tuner_config.objective_threshold:
                    print(f"There is an objective degradation, but still good..trying more aggressive knob val")
                    last_attempt_successful = True
                else:
                    print(f"This objective is not good.")
                    last_attempt_successful = False
                    
                if last_attempt_successful:
                    best_mod = self.module
                    best_state = knobs + []
                self.module = mod_bak
                intp = propagate_tensor_shapes()

                # When the knobs are set manually or no tuning, no need to explore.
                # if len(set_knobs_manually) or no_tune > 0:
                #     best_state = knobs
                #     break

                # 3. Tune the knob (TODO: this part needs exploration!)
                # If no more knobs to tune, knobs = [].
                knobs = self.tuner_config.tuner.generate_next_candidate(_match, last_attempt_successful)
                i += 1

                # TMP: early stop for debugging.
                #if i == 2:
                #    break
                if len(knobs) == 0:
                    break

            if best_mod is not None:
                verbose_print(f"Choosing the best knob module {best_state=}")
                final_knobs = best_state
                self.module = best_mod
                changed = True
                intp = propagate_tensor_shapes()

            # Run any additional non-tunable passes as needed.
            for p in list(filter(lambda p: not isinstance(p, TunablePass), self.pass_list)):
                _match = self.find_match_locations(p)
                if len(_match) == 0:
                    continue
                changed = True
                self.apply_passes([(p, m, get_inputs(m, intp)) for m in _match], [])
                intp = propagate_tensor_shapes()

        # Now we are returning a new self.module instead of directly modifying self.module
        # in-place, because there are many versions.
        # TODO: Need to check if this works with my training code well.
        if return_best_knobs:
            return self.module, final_knobs
        else:
            return self.module

    def find_match_locations(self, p):
        # Find inputs to the matched patterns and use it to
        # export replacement nodes. As the initial match
        # is done without knowing the inputs, we use
        # aten graph for pattern (instead, it is possible to
        # use dummy input + export, but we remove that for
        # simplicity.
        
        # can we cache this? It doesn't seems to be too slow anyways
        #pattern = symbolic_trace(p.get_pattern()).graph
        res = []
        patterns = [symbolic_trace(pattern).graph for pattern in p.get_patterns()]
        for pattern in patterns:
            matcher = SubgraphMatcher(
                pattern,
                match_output=False,
                match_placeholder=False,
            )
            _match = matcher.match(self.module.graph)

            # 4. Filter the matches.
            match_filters = p.get_match_filters()
            _match = [
                m
                for m in _match
                    if all(
                        #match_filter(get_head(m.nodes_map), intp.fetch_args_kwargs_from_env(m.nodes_map[get_head(m.nodes_map)])[0])
                        match_filter(m.nodes_map[get_head(m.nodes_map)])
                        for match_filter in match_filters
                    )
            ]
            res += _match

        return res

    def apply_passes(self, _match, knobs):
        pass_stat = {}

        for i, (p, match, inputs) in enumerate(_match):
            # print(f"{match=}")
            # print(f"{match.nodes_map=}")
            head = get_head(match.nodes_map)
            # print(f"{match.nodes_map[head]=}")
            # print(f"{type(match.nodes_map[head])=}")
            # print(f"{match.nodes_map[head].args=}")
            # print(f"{match.nodes_map[head].kwargs=}")
            
            # print(f"{inputs}")
            # print(f"{len(inputs)=}")
            if len(knobs) == 0:
                replacements = replace_matches(self.module, match, p.get_replacement(), inputs, str(p))
            else:
                replacements = replace_matches(self.module, match, p.get_replacement(knobs[i]), inputs, str(p))
            if "is_secret" in match.nodes_map[head].meta and match.nodes_map[head].meta["is_secret"]:
                # The node that will be replaced is the node passed to set_secret().
                # Mark the head of the replaced node as secret.
                # TODO: This does not exactly work when the head is not the secret,
                # but the replaced body is a secret.
                replacements[0].meta["is_secret"] = True
                replacements[0].meta["owner"] = match.nodes_map[head].meta["owner"]
            # TODO: TMP: This breaks the secret propagation when you use only one party (for debugging. Temporarily fixing it to 2).
            #propagate_secret(self.module, self.num_parties)
            propagate_secret(self.module, 2)
            if p not in pass_stat:
                pass_stat[p] = 0
            pass_stat[p] += len(_match)
        print(f"Pass {p!r} applied to {pass_stat[p]} locations")
    
    def print_passes(self):
        for p in self.pass_list:
            print(f"{p!r}")    


def propagate_secret(module, num_parties):
    # Propagate secret
    changed = True
    while changed:
        changed = False
        for n in module.graph.nodes:
            if "owner" not in n.meta: 
                changed = True
                n.meta["owner"] = [True] * num_parties
            for user in n.users:
                if "owner" not in user.meta: 
                    changed = True
                    user.meta["owner"] = [True] * num_parties
                # TODO: Operators that does not propagate values must not contaminate the owner metadata
                if user.op == "call_function" and user.target.__name__ == "sym_size.int":
                    continue
                new_meta = [x and y for x, y in zip(user.meta["owner"], n.meta["owner"])]
                if user.meta["owner"] != new_meta:
                    changed = True
                    user.meta["owner"] = new_meta + []

import torch.utils._pytree as pytree
import warnings

def set_secret(
    module: torch.fx.GraphModule,
    secret_tensors: Optional[Dict[str, Iterable[int]]]=None,
    output_owner =0,
    param_owner: Optional[int] =None,
    num_parties: int =2
):
    used_secrets = set()
    
    if secret_tensors is None:
        secret_tensors = {}
    
    if param_owner != None:
        for p in get_all_param_names(module):
            if p not in secret_tensors:
                secret_tensors[p] = [param_owner]
        
    
    for n in module.graph.nodes:
        if n.name in secret_tensors:
            # This is to remember which tensor was secret.
            # We should track when passes rewrite this away.
            used_secrets.add(n.name)
            n.meta["is_secret"] = True
            n.meta["owner"] = [False] * num_parties
            for i in range(num_parties):
                if i in secret_tensors[n.name]:
                    n.meta["owner"][i] = True
        if n.op == "output":
            if isinstance(output_owner, int):
                output_owners = [output_owner] * module._out_spec.num_leaves
            else:
                output_owners = list(module._out_spec.flatten_up_to(output_owner))
            n.meta["output_owners"] = output_owners
    # check that all specified secret_tensor entries are used
    for secret_tensor_name in secret_tensors:
        if secret_tensor_name not in used_secrets:
            warnings.warn(f"\"{secret_tensor_name}\" was specified as a secret but does not exist in the model. {secret_tensors=}", stacklevel=2)
    propagate_secret(module, num_parties)
    #for n in module.graph.nodes:
    #    print(n, n.meta["owner"])
    #exit(0)

def get_all_param_names(module):
    res = []
    for n in module.graph.nodes:
        if n.op == "get_attr":
            res.append(n.name)
    return res
