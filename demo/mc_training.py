from solvent.data import MCDataset
from solvent.models import MCModel
from solvent.train import MCTrainer

ROOT = 'root'
RUN_NAME = 'mc-training'
DATA_FILE = '../../nac-sampling/gen_visuals/nacs-3-bins.json'
NSTRUCTURES = 1000
NCLASSES = 3
NATOM_TYPES = 3
BATCH_SIZE = 1
NCORES = 12
SPLIT = 0.9
NATOMS = 6
MU = 0.0
STD = 1.88

ds = MCDataset(
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

model = MCModel(
    irreps_in=f'{NATOM_TYPES}x0e',
    hidden_sizes=[125, 40, 25, 15],
    natoms=NATOMS,
    nclasses=NCLASSES,
    nlayers=4,
    max_radius=4.6,
    nbasis_funcs=8,
    nradial_layers=3,
    nradial_neurons=128,
    navg_neighbors=5.0
)
print('model initialized')

trainer = MCTrainer(
    root=ROOT,
    run_name=RUN_NAME,
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    nclasses=NCLASSES,
    description='multi-class classification training',
    ncores=NCORES
)
print('trainer initialized')

print('running training')
trainer.fit()
