from abc import ABC, abstractmethod
import numpy as np


class CVModel(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def detect(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError
