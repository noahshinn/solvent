import torch
from solvent import models, train

DATA_FILE = 'new-data.json'
TRAIN_PRELOAD = './ex-preloaded-train.pt'
TEST_PRELOAD = './ex-preloaded-test.pt'
NSTRUCTURES = 10
BATCH_SIZE = 1
SPLIT = 0.9

NATOMS = 51
NATOM_TYPES = 3
NSTATES = 3


# load train and test loaders
train_loader = torch.load(TRAIN_PRELOAD)
test_loader = torch.load(TEST_PRELOAD)
print('loaders')

# initialize model
model = models.Model(
    irreps_in=f'{NATOM_TYPES}x0e',
    hidden_sizes=[125, 40, 25, 15],
    irreps_out=f'{NSTATES}x0e',
    nlayers=3,
    max_radius=4.6,
    nbasis_funcs=8,
    nradial_layers=2,
    nradial_neurons=128,
    navg_neighbors=16.0,
    cache=None
)
print('model initialized')

# initialize trainer
trainer = train.Trainer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    energy_contribution=1.0,
    force_contribution=25.0,
    description='test run'
)
print('trainer initialized')

# run training
print('running training!')
trainer.fit()
