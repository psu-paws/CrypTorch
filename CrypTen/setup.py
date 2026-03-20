#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import re
import sys

import setuptools


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "crypten"))

# Read description and requirements.
with open("README.md", encoding="utf8") as f:
    readme = f.read()
# with open("requirements.txt") as f:
#     reqs = f.read()

# get version string from module
init_path = os.path.join(os.path.dirname(__file__), "crypten/__init__.py")
with open(init_path, "r") as f:
    version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)

# Set key package information.
DISTNAME = "crypten-plus-plus"
DESCRIPTION = "CrypTen: secure machine learning in PyTorch."
LONG_DESCRIPTION = readme
AUTHOR = "Facebook AI Research"
LICENSE = "MIT licensed, as found in the LICENSE file"
REQUIREMENTS = [
    "torch>=1.7.0",
    "torchvision>=0.9.1",
    "omegaconf>=2.0.6",
    "onnx>=1.7.0",
    "pandas>=1.2.2",
    "pyyaml>=5.3.1",
    "tensorboard",
    "future",
    "scipy>=1.6.0",
    "scikit-learn",
]
VERSION = version

# Run installer.
if __name__ == "__main__":
    if sys.version_info < (3, 10):
        sys.exit("Sorry, Python >=3.10 is required for CrypTen.")

    setuptools.setup(
        name=DISTNAME,
        install_requires=REQUIREMENTS,
        packages=setuptools.find_packages(),
        dependency_links=[],
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        url="https://github.com/facebookresearch/CrypTen",
        author=AUTHOR,
        license=LICENSE,
        tests_require=["pytest"],
        package_data={"crypten": ["default.yaml"]}
        # data_files=[("configs", ["configs/default.yaml"])],
    )
