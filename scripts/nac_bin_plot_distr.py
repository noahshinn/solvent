import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

assert len(sys.argv) == 3
DATA_DIR = sys.argv[1]
SAVE_FILE = sys.argv[2]

NBINS = 10


dir_ = os.fsencode(DATA_DIR)
fig, axs = plt.subplots(len(os.listdir(dir_)))
plt.tight_layout()
for i, file in enumerate(os.listdir(dir_)):
    f = os.fsdecode(file)
    if f.endswith('.json'):
        with open(os.path.join(DATA_DIR, f), 'r') as rf:
            data = json.load(rf)
            nac_avg_elem_mag = np.absolute(np.asarray(data['nacs'])).mean(axis=(1, 2))

            target_scale = np.absolute(np.asarray(data['nacs'])).std()
            target_vals = nac_avg_elem_mag / target_scale
            mean = target_vals.mean()
            std = target_vals.std()
            axs[i].hist(target_vals, bins=NBINS, range=(mean - std * 3, mean + std * 3))
            axs[i].set_title(f)
plt.show()
fig.savefig(SAVE_FILE)
