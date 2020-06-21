import numpy as np
from sklearn.decomposition import PCA

from transform.transformer import TimeSeriesTransformer
from sklearn.preprocessing import MinMaxScaler


class ScaleTransform(TimeSeriesTransformer):

    def __init__(self, inf: float = 0, sup: float = 1):
        self.scaler = MinMaxScaler(feature_range=(inf, sup))

    def __call__(self, data: np.ndarray):
        assert data.ndim == 1
        return self.scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
