import numpy as np
from sklearn.decomposition import PCA

from transform.transformer import TimeSeriesTransformer


class PCATransformer(TimeSeriesTransformer):

    def __init__(self, n_components: int):
        self.pca = PCA(n_components=n_components)

    def __call__(self, data: np.ndarray):
        assert data.ndim == 2
        return self.pca.fit_transform(data)
