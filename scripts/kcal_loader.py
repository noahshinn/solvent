from solvent import data

DATA_FILE = 'new-data.json'
BATCH_SIZE = 1

ds = data.EnergyForceDataset(DATA_FILE, units='kcal')
ds.load()

mean_energy = ds.get_energy_mean()
rms_force = ds.get_force_rms()
ds.to_target_energy(shift_factor=mean_energy, scale_factor = 1 / rms_force)
ds.to_target_force(scale_factor = 1 / rms_force)

loader = data.DataLoader(ds.get_dataset(), batch_size=BATCH_SIZE, shuffle=True)
