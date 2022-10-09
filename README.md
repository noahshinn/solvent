# Solvent
Solvent is an open-source code for training highly accurate equivariant deep learning interatomic potentials for multiple electronic states.

**PLEASE NOTE:** the Solvent code is still under active development.

## Installation
Requires:
- Python >= 3.7

To install:
  * Create virtual environment
  ```
  python -m venv ./solvent_venv
  source ./solvent_venv/bin/activate
  ```
  * Install [torch](https://pytorch.org/) (nightly build for vmap) with CUDA
  ```
  pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu116
  ```
  * Install [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), [torch-scatter](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), [torch-cluster](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
  ```
  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
  ```
  * Install [e3nn](https://e3nn.org/)
  ```
  pip install e3nn
  ```
  * Install [joblib](https://joblib.readthedocs.io/en/latest/installing.html) for multiprocessing
  ```
  pip install joblib
  ```
  * Install [yaml](https://pypi.org/project/PyYAML/) for alternate config
  ```
  pip install pyyaml
  ```
  * Install Solvent from source by running:
  ```
  git clone https://github.com/noahshinn024/solvent.git
  cd solvent
  pip install -e .
  ```

To run demo:
  ```
  cd ./demo
  python example_training_from_preload.py
  ```

## Authors
* Noah Shinn
* Sulin Liu 
