# Solvent
Solvent is an open-source code for training E(3)-equivariant interatomic potentials.

**PLEASE NOTE:** the Solvent code is still under active development.

## Installation
Python >= 3.7
PyTorch >= 1.12+CUDA

To install:
  * Install [e3nn](https://e3nn.org/)
  ```
  pip install e3nn
  ```
  
  * Install joblib for multiprocessing
  ```
  pip install joblib
  ```

  * Install Solvent (dev)
  ```
  git clone https://github.com/noahshinn024/solvent.git
  cd solvent
  python setup.py develop
  ```
