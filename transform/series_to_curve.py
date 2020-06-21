from abc import ABC, abstractmethod

from typing import List

import numpy as np

from transform.transformer import TimeSeriesTransformer
import scipy.interpolate as interp


class IndicesWindow:

    def __init__(self, data: List[int]):
        self.data = np.array(sorted(data))

    @staticmethod
    def range(size: int, step: int):
        return IndicesWindow(list(range(0, size, step)))

    def get_indices(self, offset: int) -> np.ndarray:
        return self.data + offset

    def max_index(self) -> int:
        return self.data[-1]

    def dim(self) -> int:
        return self.data.shape[0]


class CurveProjection(TimeSeriesTransformer):

    def __init__(self, window: IndicesWindow, step: int):
        self.window = window
        self.step = step

    def __call__(self, data: np.ndarray) -> np.ndarray:
        assert data.ndim == 1
        n: int = data.shape[0]
        NWindows = int(np.ceil((n - self.window.max_index() - 1) / self.step))
        X = np.zeros((NWindows, self.window.dim()))
        i = 0
        for offset in range(0, n - self.window.max_index() - 1, self.step):
            idxx = self.window.get_indices(offset)
            data_i = data[idxx].reshape(self.window.dim())
            X[i, :] = data_i
            i += 1
        return X

