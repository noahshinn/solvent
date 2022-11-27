import sys
import torch

from solvent.data import NACDataset
from solvent.models import NACModel
from solvent.utils import renormalize, mae, mse, mape

assert len(sys.argv) == 4
DATA_FILE = sys.argv[1]
MODEL_FILE = sys.argv[2]
NSTRUCTURES = int(sys.argv[3])

NATOMS = 6
NATOM_TYPES = 3
MU = 0.0
STD = 1.88
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def rescale(x: torch.Tensor) -> torch.Tensor:
    return x * STD

ds = NACDataset(
    json_file=DATA_FILE,
    nstructures=NSTRUCTURES,
    one_hot_key={
        'H': [1., 0., 0.],
        'C': [0., 1., 0.],
        'O': [0., 0., 1.]
    },
)
ds.load()
print(f'loaded dataset: {len(ds)} structures')

model = NACModel(
    irreps_in=f'{NATOM_TYPES}x0e',
    hidden_sizes=[125, 40, 25, 15],
    natoms=NATOMS,
    nlayers=4,
    max_radius=4.6,
    nbasis_funcs=8,
    nradial_layers=3,
    nradial_neurons=128,
    navg_neighbors=5.0
)
model.load_state_dict(torch.load(MODEL_FILE, map_location='cpu')['model'])
model.eval()
model.to(DEVICE)
print(f'model loaded with {sum(p.numel() for p in model.parameters())} parameters')


c_mae = 0.0
for i, structure in enumerate(ds._dataset):
    structure.to(DEVICE)
    out = model(structure)
    pred = renormalize(out, MU, STD)
    structure['nacs'].to(DEVICE)
    print(f'prediction: {pred}')
    print(f'actual: {structure["nacs"]}')
    print(f'MAE: {mae(pred, structure["nacs"])}')
    print(f'MSE: {mse(pred, structure["nacs"])}')
    print(f'MAPE: {mape(pred, structure["nacs"])}')


