import more_itertools
from transform.transformer import TimeSeriesTransformer
import numpy as np


class TrianglePattern(TimeSeriesTransformer):

    def __init__(self, size: int):
        self.size = size

    def __call__(self, data: np.ndarray) -> np.ndarray:

        res = []

        for window in more_itertools.windowed(data, self.size, step=1):
            h = int(len(window) / 2)
            y1 = window[0:h + 1]
            y2 = np.flip(window[h:2 * h + 1])
            x = np.arange(0, h + 1) * 1.0

            y1x1 = np.dot(y1, x)
            y2x2 = np.dot(y2, x)
            norm = np.dot(x, x)

            alpha =  (y1x1 + y2x2) / norm
            y_min = np.mean(window) - alpha * len(window) / 4.0

            if alpha < 0:
                alpha = 0

            triangle = np.asarray(
                [min(i, 2 * h - i) * alpha for i in range(2 * h + 1)]
            )

            res.append(np.dot(triangle, window - y_min))

        return np.asarray(res)
