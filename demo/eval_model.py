import torch
from solvent.models import Model
from solvent.nn import force_grad
from solvent.utils import kcal_to_hartree

CHKPT_FILE = '../res/test_params.pt'
DATA_LOADER_FILE = './ex-preloaded-train.pt'
NATOM_TYPES = 3
NSTATES = 3

MEAN_ENERGY = -833713.75
RMS_FORCE = 17.097335815429688


# initialize model
model = Model(
    irreps_in=f'{NATOM_TYPES}x0e',
    hidden_sizes=[125, 40, 25, 15],
    irreps_out=f'{NSTATES}x0e',
    nlayers=4,
    max_radius=4.6,
    nbasis_funcs=8,
    nradial_layers=3,
    nradial_neurons=128,
    navg_neighbors=16.0,
    cache=None
)
print('model initialized')

# load model
model.load_state_dict(torch.load(CHKPT_FILE, map_location='cpu')['model'])
model.eval()
print('model loaded')

# initialize data loader
dl = torch.load(DATA_LOADER_FILE)

# execute model
for structure in dl:
    structure.pos.requires_grad = True
    y = model(structure)
    dy_dpos = force_grad(y, pos=structure.pos, device='cpu')
    f = dy_dpos * RMS_FORCE
    e = y * RMS_FORCE + MEAN_ENERGY
    print(kcal_to_hartree(e))
    print(kcal_to_hartree(f))
    import sys
    sys.exit(0)

