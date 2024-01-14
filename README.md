# Solvent

Solvent is an open-source code for training highly accurate equivariant deep learning interatomic potentials for multiple electronic states.

## Installation

Requires:

- Python >= 3.7
- CUDA >= 11.6

To install:

- Create virtual environment

```bash
python -m venv ./solvent_venv
source ./solvent_venv/bin/activate
```

- Install [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) 11.6

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda-repo-rhel7-11-6-local-11.6.0_510.39.01-1.x86_64.rpm
sudo rpm -i cuda-repo-rhel7-11-6-local-11.6.0_510.39.01-1.x86_64.rpm
sudo yum clean all
sudo yum -y install nvidia-driver-latest-dkms cuda
sudo yum -y install cuda-drivers
```

- Install [torch](https://pytorch.org/) with [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) 11.6

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

- Install [torch](https://pytorch.org/) - nightly build for vmap - with CUDA (optional)

```bash
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu116
```

- Install [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), [torch-scatter](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), [torch-cluster](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
```

- Install the rest of the dependencies:

```bash
pip install -r ./requirements.txt
```

To run demo:

```bash
cd ./demo
python example_training_from_preload.py
```
