import os
import sys
import json
import subprocess
import numpy as np

assert len(sys.argv) == 2
DATA_DIR = sys.argv[1]
_NBINS = int(DATA_DIR.split('/')[-1].split('-bins')[0])
_EFFECTIVE_NBINS = _NBINS - 2
ROOT = 'root-per-bin-training'
NSTRUCTURES = 100
NATOM_TYPES = 3
BATCH_SIZE = 1
NCORES = 128
SPLIT = 0.9
NATOMS = 6
MU = 0.0

def compute_std(file: str) -> float:
    with open(file, 'r') as f:
        data = json.load(f)
        nacs = np.asarray(data['nacs'])
        return float(nacs.std(axis=1))

for i in range(_EFFECTIVE_NBINS):
    _CUR_BIN = i + 1
    _RUN_NAME = f'nac-training-{_NBINS}-bins'
    _DATA_FILE = os.path.join(DATA_DIR, f'{_NBINS}-bins-{_CUR_BIN}-data.json')
    _STD = compute_std(_DATA_FILE)
    cmd = 'python ./run_single_bin_training.py'
    cmd += f' {ROOT}'
    cmd += f' {_RUN_NAME}'
    cmd += f' {_DATA_FILE}'
    cmd += f' {NSTRUCTURES}'
    cmd += f' {NATOM_TYPES}'
    cmd += f' {BATCH_SIZE}'
    cmd += f' {NCORES}'
    cmd += f' {SPLIT}'
    cmd += f' {NATOMS}'
    cmd += f' {MU}'
    cmd += f' {_STD}'
    cmd += f' {_NBINS}'
    cmd += f' {_CUR_BIN}'

    submission_file = f'{_NBINS}-bins-{_CUR_BIN}.slurm'
    slurm_str = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=128Gb
#SBATCH --cpus-per-task={NCORES}
#SBATCH --time=23:59:00
#SBATCH --job-name=workspace
#SBATCH --partition=short


"""
    load_conda_str = 'module load anaconda3/2022.01'
    load_env_str = 'source activate e3nn_env'
    with open(submission_file, 'w') as f:
        f.write(f'{slurm_str}\n\n{load_conda_str}\n{load_env_str}\n\n{cmd}')
    subprocess.run(['sbatch', submission_file])
    print(f'submitted job #{i}')
