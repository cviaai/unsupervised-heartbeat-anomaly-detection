import ot

import numpy as np
from scipy.stats import wasserstein_distance
from statistic.two_sample_statistic import TwoSampleStatistic


class WassersteinDistance1D(TwoSampleStatistic):

    @staticmethod
    def to_hist(img: np.ndarray):
        img = (img * 100).astype(np.int)
        h, w = img.shape
        hist = [0.0] * 1000
        for i in range(h):
            for j in range(w):
                hist[img[i, j]] += 1
        result = np.array(hist) / (h * w)
        return result

    def __call__(self, first: np.ndarray, second: np.ndarray) -> float:

        cloud_a = self.to_hist(first)
        cloud_b = self.to_hist(second)

        return wasserstein_distance(cloud_a, cloud_b)


class WassersteinDistance(TwoSampleStatistic):

    def __init__(self, p: float = 2.0):
        self.p = p

    def __call__(self, first: np.ndarray, second: np.ndarray) -> float:
        n1, n2 = first.shape[0], second.shape[0]

        p1, p2 = np.ones((n1,)) / n1, np.ones((n2,)) / n2

        return np.power(self.compute(first, second, p1, p2), 1 / self.p)

    def compute(self, first: np.ndarray, second: np.ndarray, p1, p2) -> float:
        cost: np.ndarray = np.power(
            ot.dist(first, second),
            self.p / 2
        )
        max_dist = cost.max()

        p12 = ot.sinkhorn(p1, p2, cost / max_dist, reg=0.003, numItermax=300, stopThr=1e-4)

        return cost.reshape(-1).dot(p12.reshape(-1))


class WassersteinDistanceDeviation:

    def __init__(self, p: float = 2.0):
        self.p = p
        self.dist = WassersteinDistance(p)

    def __call__(self, first: np.ndarray, second: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> float:
        assert (len(first) == len(w1))
        assert (len(second) == len(w2))

        n1, n2 = first.shape[0], second.shape[0]

        p1, p2 = np.ones((n1,)) / n1, np.ones((n2,)) / n2
        p1w, p2w = w1 / w1.sum(), w2 / w2.sum()

        dwas = self.dist.compute(first, first, p1, p1w) + \
               self.dist.compute(second, second, p2, p2w)

        return np.power(np.abs(dwas), 1 / self.p)



