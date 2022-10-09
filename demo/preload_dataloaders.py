import torch

from solvent import data

DATA_FILE = 'new-data.json'
TRAIN_SAVE_FILE = 'ex-preloaded-train.pt'
TEST_SAVE_FILE = 'ex-preloaded-test.pt'
NSTRUCTURES = 10
BATCH_SIZE = 1
SPLIT = 0.9

NATOMS = 51
NSTATES = 3


# load dataset from json
ds = data.EnergyForceDataset(
    json_file=DATA_FILE,
    nstructures=NSTRUCTURES,
    one_hot_key={
        'H': [1., 0., 0.],
        'C': [0., 1., 0.],
        'O': [0., 0., 1.]
    },
    units='kcal')
ds.load()
print('loaded dataset')

# compute constants for target data shifting and scaling
mean_energy = ds.get_energy_mean()
rms_force = ds.get_force_rms()
ds.to_target_energy(shift_factor=mean_energy, scale_factor = 1 / rms_force)
ds.to_target_force(scale_factor = 1 / rms_force)
print('constants computed')

# train and test loaders
train_loader, test_loader = ds.gen_dataloaders(
    split=SPLIT,
    batch_size=BATCH_SIZE,
    should_shuffle=True
)

# save loaders
torch.save(train_loader, TRAIN_SAVE_FILE)
torch.save(test_loader, TEST_SAVE_FILE)
print('loaders saved!')
