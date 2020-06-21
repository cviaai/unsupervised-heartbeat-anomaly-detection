from typing import TypeVar, Iterator, Tuple, List, Callable

import biosppy
import numpy as np

from transform.interpolate import SplineInterpolate
from transform.scale import ScaleTransform
from transform.transformer import TimeSeriesTransformer, SequentialTransformer
import itertools
import more_itertools


class RPeaksFinder(TimeSeriesTransformer):

    def __call__(self, data: np.ndarray):
        peaks = biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate=200)[0]
        return peaks


T = TypeVar('T')


class ECGSlider:

    def __init__(self, data: np.ndarray, periods_in_window: int, padding: int = 0):
        self.data = data
        self.size = periods_in_window 
        self.padding = padding

        transform = SequentialTransformer(
            ScaleTransform(0, 1),
            SplineInterpolate(0.01),
            RPeaksFinder()
        )

        self.max_indices = transform(data)

    def iterator(self) -> Iterator[Tuple[List[int], np.ndarray]]:
        for index_wd in more_itertools.windowed(self.max_indices, self.size, step=1):
            index_wd = list(index_wd)
            i1 = max(0, index_wd[0] - self.padding)
            i2 = min(self.data.shape[0], index_wd[-1] + self.padding)
            yield list(range(i1, i2)), self.data[i1:i2].copy()

    def map(self, f: Callable[[np.ndarray], T]) -> List[Tuple[int, T]]:
        return [(int((index[0] + index[-1]) / 2), f(window_data)) for index, window_data in self.iterator()]

