#!/bin/sh

#wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda-repo-rhel7-11-6-local-11.6.0_510.39.01-1.x86_64.rpm
#sudo rpm -i cuda-repo-rhel7-11-6-local-11.6.0_510.39.01-1.x86_64.rpm
#sudo yum clean all
#sudo yum -y install nvidia-driver-latest-dkms cuda
#sudo yum -y install cuda-drivers

# sudo apt-get update && sudo apt-get dist-upgrade
#python -m venv ./venv
#source ./venv/bin/activate

# torch - ML framework
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# torch-geometric, torch-scatter, torch-sparse, torch-cluster, torch-spline-conv - neural network math
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html

# e3nn - neural network math
pip install e3nn

# joblib - multiprocessing
pip install joblib

# yaml loading
pip install pyyaml
