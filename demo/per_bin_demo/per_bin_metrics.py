import os
import sys
import json
import subprocess
import numpy as np

assert len(sys.argv) == 2
DATA_DIR = sys.argv[1]
if DATA_DIR.endswith('/'):
    _NBINS = int(os.path.basename(DATA_DIR[:-1]).split('-')[0])
else:
    _NBINS = int(os.path.basename(DATA_DIR).split('-')[0])
_EFFECTIVE_NBINS = _NBINS - 2
ROOT = 'root-per-bin-training'
NSTRUCTURES = 1000
NATOM_TYPES = 3
BATCH_SIZE = 1
NCORES = 128
SPLIT = 0.9
NATOMS = 6
MU = 0.0
USE_RESERVATION = True

def compute_std_mag(data: dict) -> float:
    nacs = np.asarray(data['nacs'])
    out = float(np.absolute(nacs).std())
    return out

def compute_median_mag(data: dict) -> float:
    nacs = np.asarray(data['nacs'])
    out = float(np.median(np.absolute(nacs)))
    return out

def compute_avg_mag(data: dict) -> float:
    nacs = np.asarray(data['nacs'])
    out = float(np.absolute(nacs).mean())
    return out

for i in range(_EFFECTIVE_NBINS):
    _CUR_BIN = i + 1
    _DATA_FILE = os.path.join(DATA_DIR, f'{_NBINS}-bins-{_CUR_BIN}-data.json')
    with open(_DATA_FILE, 'r') as f:
        data = json.load(f)
        std_mag = compute_std_mag(data)
        median_mag = compute_median_mag(data)
        avg_mag = compute_avg_mag(data)
        print(f"""median mag: {median_mag}
avg mag: {avg_mag}
std mag: {std_mag}

""")
