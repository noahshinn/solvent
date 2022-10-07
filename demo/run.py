import yaml
import torch

from solvent import models, utils, train

_INPUT = 'constants.yaml'

d = utils.read_yaml(_INPUT)

model = models.Model(*args) # type: ignore
train_loader = ...
test_loader = ...

trainer = train.Trainer(model, train_loader, test_loader) # type: ignore
trainer.fit()
