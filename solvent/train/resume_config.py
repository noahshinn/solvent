"""
STATUS: DEV

optional separate files?

"""

import torch
from torch.optim import (
    Adam,
    SGD
)
from torch.optim.lr_scheduler import (
    ExponentialLR,
    ReduceLROnPlateau    
)

from typing import Dict, TypeVar, Type, Union

T = TypeVar('T', bound='ResumeConfig')

# *default key
# maps target keys to keys in given save_file
_KEY = {
    'model': 'model',
    'optim': 'optim',
    'scheduler': 'scheduler',
    'epoch': 'epoch'
}

class ResumeConfig:
    """
    Usage:

    >>> from solvent import train
    >>> model = EModel(*args, **kwargs) 
    >>> optim = Optim(*args, **kwargs)
    >>> scheduler = Scheduler(*args, **kwargs)
    >>> resume_config = train.ResumeConfig.deserialize(
    ...     model=model,
    ...     optim=optim,
    ...     scheduler=scheduler,
    ...     chkpt_file='chkpt.pt',
    ... )
    >>> trainer = train.Trainer(
    ...     model=resume_config.model,
    ...     optim=resume_config.optim,
    ...     scheduler=resume_config.scheduler,
    ...     *args,
    ...     **kwargs,
    ... )
    >>> trainer.train()

    """
    def __init__(
            self,
            model: torch.nn.Module,
            optim: Union[Adam, SGD],
            scheduler: Union[ExponentialLR, ReduceLROnPlateau],
            model_state_dict: Dict,
            optim_state_dict: Dict,
            scheduler_state_dict: Dict,
            epoch: int
        ) -> None:
        """
        Args:
            model (torch.nn.Module): Model architecture.
            optim (Union[Adam, SGD]): Optimizer architecture.
            scheduler (Union[ExponentialLR, ReduceLROnPlateau]): Scheduler architecture.
            model_state_dict (Dict): Saved model parameters.
            optim_state_dict (Dict): Saved optimizer parameters.
            scheduler_state_dict (Dict): Saved scheduler parameters.
            epoch (int): The last epoch before training terminated.

        Returns:
            (None)

        """
        self.model = model.load_state_dict(model_state_dict)
        self.model.train() # FIXME: type error
        self.optim = optim.load_state_dict(optim_state_dict)
        self.scheduler = scheduler.load_state_dict(scheduler_state_dict)
        self.epoch = epoch + 1
    
    @classmethod
    def deserialize(
            cls: Type[T],
            model: torch.nn.Module,
            optim: Union[Adam, SGD],
            scheduler: Union[ExponentialLR, ReduceLROnPlateau],
            chkpt_file: str,
            key: Dict = _KEY
        ) -> T:
        """
        Deserializes checkpoint file.

        Args:
            model (torch.nn.Module): Model architecture.
            optim (Union[Adam, SGD]): Optimizer architecture.
            scheduler (Union[ExponentialLR, ReduceLROnPlateau]): Scheduler architecture.
            chkpt_file (str): Checkpoint file.
            key (Dict): Decode key for the checkpoint file.
                *default: 
                    key = {
                        'model': 'model',
                        'optim': 'optim',
                        'scheduler': 'scheduler',
                        'epoch': 'epoch'
                    }

        Returns:
            (ResumeConfig): An instance of ResumeConfig.

        Asserts:
            - checkpoint file extension is .pt

        """
        assert chkpt_file.endswith('.pt')
        d = torch.load(chkpt_file)
        return cls(
            model=model,
            optim=optim,
            scheduler=scheduler,
            model_state_dict=d[key['model']],
            optim_state_dict=d[key['optim']],
            scheduler_state_dict=d[key['scheduler']],
            epoch=d[key['epoch']],
        )
