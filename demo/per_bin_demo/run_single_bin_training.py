import sys
from solvent.data import NACDataset
from solvent.models import NACModel 
from solvent.train import NACTrainer


assert len(sys.argv) == 12
ROOT = sys.argv[1]
RUN_NAME = sys.argv[2]
DATA_FILE = sys.argv[3]
NSTRUCTURES = int(sys.argv[4])
NATOM_TYPES = int(sys.argv[5])
BATCH_SIZE = int(sys.argv[6])
NCORES = int(sys.argv[7])
SPLIT = float(sys.argv[8])
NATOMS = int(sys.argv[9])
MU = float(sys.argv[10])
STD = float(sys.argv[11])
NBINS = int(sys.argv[12])
CUR_BIN = int(sys.argv[13])

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

train_loader, test_loader = ds.gen_dataloaders(
    split=SPLIT,
    batch_size=BATCH_SIZE,
    should_shuffle=True
)
print('loaders')

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
print('model initialized')

trainer = NACTrainer(
    root=ROOT,
    run_name=RUN_NAME,
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    mu=MU,
    std=STD,
    description=f'{NBINS} bin nac inference training for bin #{CUR_BIN}',
    ncores=NCORES
)
print('trainer initialized')

print(f'running training for `{NBINS} bin nac inference training for bin #{CUR_BIN}`')
trainer.fit()
