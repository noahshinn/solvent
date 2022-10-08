from solvent import data

DATA_FILE = 'new-data.json'
BATCH_SIZE = 1

ds = data.EnergyForceDataset(DATA_FILE, nstructures=10, units='kcal')
ds.load()

mean_energy = ds.get_energy_mean()
rms_force = ds.get_force_rms()
ds.to_target_energy(shift_factor=mean_energy, scale_factor = 1 / rms_force)
ds.to_target_force(scale_factor = 1 / rms_force)

train_loader, test_loader = ds.gen_dataloaders(
    split=0.9,
    batch_size=1,
    should_shuffle=True
)

print('train loader')
for structure in train_loader:
    print(structure)

print('test loader')
for structure in test_loader:
    print(structure)
