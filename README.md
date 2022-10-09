# Solvent
Solvent is an open-source code for training highly accurate equivariant deep learning interatomic potentials for multiple electronic states.

**PLEASE NOTE:** the Solvent code is still under active development.

## Installation
Requires:
- Python >= 3.7

To install:
  * Init virtual env
  ```
  python -m venv ./venv
  source ./venv/bin/activate
  ```
  * Install torch with CUDA
  ```
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
  ```
  * Install torch-geometric, torch-scatter, torch-cluster
  ```
  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
  ```
  * Install [e3nn](https://e3nn.org/)
  ```
  pip install e3nn
  ```
  * Install joblib for multiprocessing
  ```
  pip install joblib
  ```
  * Install yaml for alternate config
  ```
  pip install pyyaml
  ```
  * Install Solvent from source by running:
  ```
  git clone https://github.com/noahshinn024/solvent.git
  cd solvent
  pip install -e .
  ```

## Authors
* Noah Shinn
* Sulin Liu 
