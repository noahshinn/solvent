import abc


class LossMixin:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        """execute"""
        return
    
    @abc.abstractmethod
    def reset(self, *args, **kwargs) -> None:
        """reset after epoch"""
        return
