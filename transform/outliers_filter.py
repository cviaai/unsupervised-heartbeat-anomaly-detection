

from typing import Tuple, List

from transform.transformer import TimeSeriesTransformer
import numpy as np


class OutliersFilter:

    def __init__(self, sigma_mult: float = 3.0):
        self.sigma_mult = sigma_mult

    def __call__(self, data: np.ndarray) -> Tuple[List[int], np.ndarray]:

        m = data.mean()
        x = data - m
        sigma = np.sqrt(np.power(x, 2).sum())
        data = data[data < m + self.sigma_mult * sigma]
        data = data[data > m - self.sigma_mult * sigma]

        return data
