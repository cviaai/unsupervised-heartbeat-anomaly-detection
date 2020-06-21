import pandas as pd
import numpy as np
import sys,os
import time
import biosppy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy

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
from func.Functions import std_mean_change
from tqdm import tqdm 

#here additional functions located
dist = WassersteinDistance()


def wasserstein_computation(smooth_data,size,p1,p2,n_repeats,periods=10,padding=10):
    slider = ECGSlider(smooth_data, periods,padding).iterator()
    dist = WassersteinDistance(p1)
    dist_dev = WassersteinDistanceDeviation(p2)
    projection_step=1
    curve_transform = SequentialTransformer(
        CurveProjection(
            window=IndicesWindow.range(size=size, step=2),
            step=projection_step
        ),
        PCATransformer(10)
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


def statistic(was_index,was,was_deviation_median,sig_series):
    new_data=std_mean_change(was_deviation_median,was)
    line=np.quantile(new_data,0.95)
    sep_line1=[line]*len(sig_series)

    triangle = IndexedTransformer(TrianglePattern(7), padding=1, step=1)
    tr_indices, tr_was = triangle(np.asarray(was))
    tr_indices_dev, tr_was_dev = triangle(np.asarray(new_data))
    final_indices = np.asarray(was_index)[tr_indices]

    f_i_d=np.array([])
    ind=np.array([])
    for i,j in enumerate(tr_was):
        #if max(tr_was)>=min(tr_was):
        if j>line:
            f_i_d=np.append(j,f_i_d)
            ind=np.append(i,ind)
    fin_ind=np.array([])
    for i in ind:
        fin_ind=np.append(final_indices[int(i)],fin_ind)

    f_i_d =f_i_d[::-1]
    return sep_line1,f_i_d,final_indices,tr_was,fin_ind