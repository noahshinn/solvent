import torch

from solvent import models, train

DATA_FILE = './new-data.json'
TRAIN_PRELOAD = './ex-preloaded-train.pt'
TEST_PRELOAD = './ex-preloaded-test.pt'
CHKPT_FILE = './ex-chkpt.pt'
NSTRUCTURES = 10
BATCH_SIZE = 1
SPLIT = 0.9

NATOMS = 51
NSTATES = 3


# initialize model
model = models.Model(
    irreps_in=f'{NATOMS}x0e',
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

# initialize optimizer and scheduler
optim = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)

# resume config
resume_config = train.ResumeConfig.deserialize(
    model=model,
    optim=optim,
    scheduler=scheduler,
    chkpt_file=CHKPT_FILE
)

# load train and test loaders
train_loader = torch.load(TRAIN_PRELOAD)
test_loader = torch.load(TEST_PRELOAD)
print('loaders')

# initialize trainer
trainer = train.Trainer(
    model=resume_config.model, # FIXME: type error
    train_loader=train_loader,
    test_loader=test_loader,
    optim=resume_config.optim,
    scheduler=resume_config.scheduler,
    energy_contribution=1.0,
    force_contribution=25.0,
    start_epoch=resume_config.epoch,
    description='test resume run'
)
print('trainer initialized')

# run training
print('running training!')
trainer.fit()
