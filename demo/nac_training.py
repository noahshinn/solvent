from solvent.data import NACDataset
from solvent.models import BinModel
from solvent.train import BinTrainer

ROOT = 'root'
RUN_NAME = 'nac-training'
DATA_FILE = '../../nac-sampling/data-nacs.json'
NSTRUCTURES = 100
BATCH_SIZE = 1
SPLIT = 0.9

NATOMS = 6

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
print('loaded dataset')

train_loader, test_loader = ds.gen_dataloaders(
    split=SPLIT,
    batch_size=BATCH_SIZE,
    should_shuffle=True
)
print('loaders')

model = BinModel(
    irreps_in=f'{NATOM_TYPES}x0e',
    hidden_sizes=[125, 40, 25, 15],
    nlayers=4,
    max_radius=4.6,
    nbasis_funcs=8,
    nradial_layers=3,
    nradial_neurons=128,
    navg_neighbors=5.0,
    cache=None
)
print('model initialized')

trainer = BinTrainer(
    root=ROOT,
    run_name=RUN_NAME,
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    description='nac classification model'
)
print('trainer initialized')

print('running training')
trainer.fit()
