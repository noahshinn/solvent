# Solvent
Solvent is an open-source code for training highly accurate equivariant deep learning interatomic potentials for multiple electronic states.

**PLEASE NOTE:** the Solvent code is still under active development.

## Installation
Python >= 3.7
torch >= 1.12+CUDA

To install:
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

  * Install Solvent from source by running:
  ```
  git clone https://github.com/noahshinn024/solvent.git
  cd solvent
  python setup.py develop
  ```

## Authors
* Noah Shinn
* Sulin Liu 
