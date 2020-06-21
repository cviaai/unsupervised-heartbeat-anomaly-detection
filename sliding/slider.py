import itertools
import more_itertools

import numpy as np
from typing import Iterator, TypeVar, Callable, List

from joblib import Parallel, delayed
import multiprocessing

T = TypeVar('T')


class Slider:

    def __init__(self, data: np.ndarray, size: int, step: int = 1):
        self.data = data
        self.size = size
        self.step = step
        self.par = Parallel(n_jobs=multiprocessing.cpu_count())

    def iterator(self) -> Iterator[np.ndarray]:
        n = self.data.shape[0]
        new_n = int(np.floor((n - self.size) / self.step)) * self.step + self.size

        for wd in more_itertools.windowed(self.data[0:new_n], self.size, step=self.step):
            yield np.asarray(list(wd))

    def map(self, f: Callable[[np.ndarray], T]) -> List[T]:
        return [f(window_data) for window_data in self.iterator()]

    def par_map(self, f: Callable[[np.ndarray], T]) -> List[T]:
        res = self.par(delayed(f)(window_data) for window_data in self.iterator())
        return res
