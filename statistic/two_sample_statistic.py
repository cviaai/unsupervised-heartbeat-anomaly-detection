from abc import ABC, abstractmethod
import numpy as np


class TwoSampleStatistic(ABC):

    @abstractmethod
    def __call__(self, first: np.ndarray, second: np.ndarray) -> float: pass
