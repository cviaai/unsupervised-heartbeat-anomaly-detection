from typing import Tuple, List

from transform.transformer import TimeSeriesTransformer
import numpy as np


class IndexedTransformer:

    def __init__(self, transformer: TimeSeriesTransformer, padding: int, step: int):
        self.transformer = transformer
        self.padding = padding
        self.step = step

    def __call__(self, data: np.ndarray) -> Tuple[List[int], np.ndarray]:
        tr_data = self.transformer(data)
        indices = [self.padding + i * self.step for i in range(len(tr_data))]
        return indices, tr_data
