<p align="center"><img width="80%" src="https://www.semicolontransistor.net/cryptorch/CrypTorchLogoText.png" alt="CrypTorch Logo" /></p>

----------


CrypTorch is an **MPC-based ML framework** built on top of PyTorch, developed and maintained by the [PAWS lab](https://github.com/psu-paws) @Penn State. CrypTorch enables you to run your **PyTorch-written ML program** using **multi-party computing (MPC)**, allowing model training and inference while ensuring the confidentiality of input data and model weights. 

Description of the framework can be found in our [arxiv paper](https://arxiv.org/abs/2511.19711).

# Setting up CrypTorch

1. Clone this repo
    ```
    git clone https://github.com/psu-paws/CrypTorch.git
    cd CrypTorch
    ```
    
2. Setup a virtual environment (Optional)

    We recommend setting up and activating a virtual environment ([venv](https://docs.python.org/3/library/venv.html), [conda](https://anaconda.org/), etc) to avoid installing packages globally and potentially causing conflicts. 

    * Conda (Example)
        ```
        conda create -n CrypTorch python=3.12
        conda activate CrypTorch
        ```
    
    * venv (Example)
        ```
        python3.12 -m venv .venv
        source .venv/bin/activate
        ```
3. Install dependencies

    1. Install PyTorch version 2.8.0 using instructions from PyTorch's [website](https://pytorch.org/get-started/previous-versions/). For example, to install CPU-only version of PyTorch, use the following command. **If you have GPUs, install a version of PyTorch 2.8.0 that is compatible with your cuda version instead**.
        ```
        pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
        ```

    2. Install crypten-plus-plus (modified CrypTen)

        ```
        pip install -e ./CrypTen
        ```
    
    3. Install cryptorch
        ```
        pip install -e .
        ```

4. (When using GPU) Build GPU integer kernels
    Follow instructions under `cutlass/README.md`.

# User Slack channel

To join our user Slack channel, use the following [link](https://join.slack.com/t/cryptorchusers/shared_invite/zt-3t1mjzide-tXQi_mmoyx193OW9rMkVhg).

# Citing CrypTorch

To cite our work, use the below Bibtex:
```
@article{liu2025cryptorch,
  title={CrypTorch: PyTorch-based Auto-tuning Compiler for Machine Learning with Multi-party Computation},
  author={Liu, Jinyu and Tan, Gang and Maeng, Kiwan},
  journal={arXiv preprint arXiv:2511.19711},
  year={2025}
}
```
