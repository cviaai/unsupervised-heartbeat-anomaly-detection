from abc import abstractmethod, ABC

import numpy as np
from typing import List


class TimeSeriesTransformer(ABC):

    @abstractmethod
    def __call__(self, data: np.ndarray): pass


class SequentialTransformer(TimeSeriesTransformer):

    def __init__(self, *transformers: TimeSeriesTransformer):
        self.transformers = transformers

    def __call__(self, data: np.ndarray) -> np.ndarray:
        tmp = data
        for transform in self.transformers:
            tmp = transform(tmp)
        return tmp
