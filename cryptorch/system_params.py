# TODO: TMP: Hardcoded for now

import collections.abc
import os
import typing
import pathlib

import yaml
try:
    from yaml import CLoader as YAMLLoader, CDumper as YAMLDumper
except ImportError:
    from yaml import YAMLLoader, YAMLDumper


_system_config = {
}

# log_encoding_scale = 16
# encoding_scale = 2 ** log_encoding_scale

# # TODO: TMP: Hardcoded for now
# ring_size = 64
# if ring_size == 64:
#     ring_dtype = "long"
# elif ring_size == 32:
#     ring_dtype = "int"
# else:
#     raise AssertionError()


def load_config(config: typing.Mapping | str | os.PathLike[str] | typing.TextIO):
    global _system_config 
    if isinstance(config, (str, os.PathLike)):
        with pathlib.Path(config).open("r") as config_file:
            _system_config = yaml.load(config_file, Loader=YAMLLoader)
            return
    
    elif isinstance(config, collections.abc.Mapping):
        _system_config = dict(config)
    
    else:
        # try to load as stream
        _system_config = yaml.load(config, loader=YAMLLoader)

def print_config():
    global _system_config
    print(yaml.dump(_system_config))

def get_config_value(path: str, default: typing.Optional[typing.Any] = None):
    global _system_config 
    def _helper(path_elements: typing.List[str], config_object: typing.Mapping, default: typing.Optional[typing.Any] = None):
        key = path_elements[0]
        remainder = path_elements[1:]
        
        if key in config_object:
            new_object = config_object[key]
            
            if remainder:
                return _helper(remainder, new_object, default)
            else:
                return new_object
        else:
            return default
    
    path_elements = path.split(".")
    return _helper(path_elements, _system_config, default)

# replacement for hard coded values
# def __getattr__(name: str):
#     if name == "encoding_scale":
#         return get_config_value("ring.encoding_scale", 65536)
#     elif name == "log_encoding_scale":
#         encoding_scale = get_config_value("ring.encoding_scale", 65536)
#         if encoding_scale.bit_count() != 1:
#             raise RuntimeError(f"Attempted to access log_encoding_scale while encoding_scale is not a power of 2. {encoding_scale=}")
#         return (encoding_scale - 1).bit_length()
#     elif name == "ring_size":
#         return get_config_value("ring.size", 64)
#     elif name == "ring_dtype":
#         ring_size = get_config_value("ring.size", 64)
#         if ring_size == 64:
#             return "long"
#         elif ring_size == 32:
#             return "int"
#         else:
#             raise RuntimeError(f"Unsupported Ring Size: {ring_size=}")
#     else:
#         raise AttributeError()

def ring_size() -> int:
    rs = get_config_value("ring.size", 64)
    return rs

def ring_dtype() -> str:
    ring_size = get_config_value("ring.size", 64)
    if ring_size == 64:
        return "long"
    elif ring_size == 32:
        return "int"
    else:
        raise RuntimeError(f"Unsupported Ring Size: {ring_size=}")

def encoding_scale() -> int:
    return get_config_value("ring.encoding_scale", 65536)

def log_encoding_scale() -> int:
    encoding_scale = get_config_value("ring.encoding_scale", 65536)
    if encoding_scale.bit_count() != 1:
        raise RuntimeError(f"Attempted to access log_encoding_scale while encoding_scale is not a power of 2. {encoding_scale=}")
    return (encoding_scale - 1).bit_length()