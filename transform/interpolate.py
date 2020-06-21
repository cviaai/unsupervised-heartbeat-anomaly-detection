import numpy as np

from transform.transformer import TimeSeriesTransformer
import scipy.interpolate as inter


class SplineInterpolate(TimeSeriesTransformer):

    def __init__(self, smooth: float):
        self.smooth = smooth

    def interpolate(self, data: np.ndarray):
        x = np.arange(data.shape[0])
        spl = inter.UnivariateSpline(x, data, k=3)
        spl.set_smoothing_factor(self.smooth)
        return spl(x)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        assert data.ndim == 1
        n_split = 1 if data.shape[0] <= 2000 else int(np.floor(data.shape[0] / 1000))
        return np.concatenate([
          self.interpolate(part) for part in np.array_split(data, n_split)
        ])
