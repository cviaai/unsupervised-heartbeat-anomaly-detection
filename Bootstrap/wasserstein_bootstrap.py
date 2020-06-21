from sliding.ecg_slider import ECGSlider
from sliding.slider import Slider
from statistic.wasserstein_distance import WassersteinDistance, WassersteinDistanceDeviation
from transform.indexed_transform import IndexedTransformer
from transform.interpolate import SplineInterpolate
from transform.pca import PCATransformer
from transform.scale import ScaleTransform
from transform.series_to_curve import CurveProjection, IndicesWindow
from transform.transformer import SequentialTransformer
from transform.triangle_pattern import TrianglePattern
from tqdm import tqdm
import ot
import numpy as np
#function calculate wasserstein distance and bootstrap
#smooth_data: input data(time series)
#p1 and p2 : p-parametr in wasserstein distance
#n_repeats:bootstrap repeat
def wasserstein_computation(smooth_data,size,p1,p2,n_repeats,n_components):
    slider = ECGSlider(smooth_data, 4,10).iterator()
    dist = WassersteinDistance(p1)
    dist_dev = WassersteinDistanceDeviation(p2)
    projection_step=1
    curve_transform = SequentialTransformer(
        CurveProjection(
            window=IndicesWindow.range(size=size, step=2),
            step=projection_step
        ),
        PCATransformer(n_components)
    )
    was=[]
    
    was_deviation_median=[]
    was_index=[]
    curves=[]
    for index, window_data in tqdm(slider):

        was_deviation_wind = []
        window_curve = curve_transform(window_data)
        curves.append(window_curve)
        h = len(window_curve) // 2
        h2 = len(window_curve)
        was_i = dist(window_curve[0:h], window_curve[h:h2])
        was_index_i = (index[0] + index[-1]) // 2
        for i in range(n_repeats):
            rand = np.random.normal(1.0, 1.0, len(smooth_data) // 20)
            weights = np.asarray([rand[i//20] for i in range(len(smooth_data))])
            w_i = weights[index]
            was_dev_i = dist_dev(window_curve[0:h], window_curve[h:h2], w_i[0:h], w_i[h:h2])
            was_deviation_wind.append(max(0, was_dev_i))

        was.append(max(0,was_i))
        was_deviation_median.append(np.median(was_deviation_wind))
        was_index.append(np.max(was_index_i))
    return was,was_deviation_median,was_index,curves