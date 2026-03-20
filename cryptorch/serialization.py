import torch
import yaml
try:
    from yaml import CLoader as YAMLLoader
except ImportError:
    from yaml import YAMLLoader

from typing import Iterable, Optional, Tuple, Any

def search_node_source(node_source, name, graph_id):

    if node_source.node_info is not None:
        node_info = node_source.node_info
        if node_info.name == name and node_info.graph_id == graph_id:
            return True
    
    for ns in node_source.from_node:
        if search_node_source(ns, name, graph_id):
            return True
    
    return False

def export_and_save(f, module, examples: Optional[Tuple[Any]] = None, additional_keys: Iterable[str] = (), **export_kwargs):
    
    if examples is None:
        examples = []
        
        for node in module.graph.nodes:
            if node.op == "placeholder":
                tensor_meta = node.meta["tensor_meta"]
                # print(tensor_meta)
                dummy_input = torch.zeros(*tensor_meta.shape, dtype=tensor_meta.dtype)
                examples.append(dummy_input)
        examples = tuple(examples)
    
    cryptorch_meta_keys = (
        "passes", "output_of", "owner", "max_abs", "output_owners"
    )
    
    cryptorch_meta_keys = cryptorch_meta_keys + additional_keys
    
    ep = torch.export.export(module, args=examples, **export_kwargs)
    
    # print(ep.graph_module)
    
    new_gm = ep.module()
        
            
    cryptorch_meta = dict()
    for node in module.graph.nodes:
        if node.op == "placeholder" or node.op == "get_attr" or node.op == "output":
            name = node.name
        elif node.op == "call_function":
            name = ""
            for n in new_gm.graph.nodes:
                if "from_node" in n.meta:
                    for ns in n.meta["from_node"]:
                        if search_node_source(ns, node.name, id(node.graph)):
                            name = n.name
                            break
                
                if name:
                    break
            if not name:
                raise RuntimeError(f"Unable to find matching node for {node}")
        else:
            assert False
        
        cryptorch_meta[name] = {}
        
        for key in cryptorch_meta_keys:
            if key in node.meta:
                cryptorch_meta[name][key] = node.meta[key]
        
    cryptorch_meta = yaml.dump(cryptorch_meta)
    
    torch.export.save(
        ep,
        f,
        extra_files={
            "cryptorch_meta": cryptorch_meta
        }
    )

def load_from(f):
    extra_files = {"cryptorch_meta": None}
    ep = torch.export.load(
        f,
        extra_files=extra_files
    )
    module = ep.module()
    cryptorch_meta = yaml.load(extra_files["cryptorch_meta"], Loader=YAMLLoader)
    # print(cryptorch_meta)
    # print(module)
    
    for node in module.graph.nodes:
        if node.name in cryptorch_meta:
            for key, value in cryptorch_meta[node.name].items():
                node.meta[key] = value
    
    return module