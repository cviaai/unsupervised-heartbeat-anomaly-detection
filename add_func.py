import time
import biosppy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import lfilter
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm 

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
from Bootstrap.wasserstein_bootstrap import wasserstein_computation
from func.Functions import chunkIt
from func.Functions import std_mean_change
from func.Functions import index_to_series
from func.Functions import annotations
from func.Functions import recall
from func.Functions import accuracy
from func.Functions import specifity
from func.Functions import  true_labels 
from func.Functions import  arrhythmia_index 
from func.Functions import  arrhythmia_index_check 
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch
import pickle
#import ot
from abc import ABC, abstractmethod

def shape_changes(curves):
    new_curves=[]
    for i in range(len(curves)):
        new=np.zeros((2236,3))
        new[:curves[i].shape[0],:3]=curves[i]
        new_curves.append(new)
    return new_curves

def features_vectors_(curves,model,i,tensor_set):   
    new_curves=shape_changes(curves)
    tensor_set=torch.stack([torch.Tensor(i) for i in new_curves])
    #preds=[]
    features=[]
    for curve in tqdm(tensor_set):
        curve=curve.reshape(1,-1)
        inp=Variable(curve)
        #print(inp)
        feature_vector=model(inp)[i]
       # prediction=prediction.reshape(2236,3).data.cpu().numpy()
        feature_vector=feature_vector.data.cpu().numpy()
        #preds.append(prediction)
        features.append(feature_vector)
    return features
def metric(a,b):
    return np.linalg.norm((a-b),2)
def predict(centers, points):
        nppoints = np.array(points)

        differences = np.zeros((len(centers),2 ))
        for i in range(len(centers)):
            differences[i] = metric(points, centers[i])
        prediction=np.argmin(differences, axis=0)[0]

        return prediction
    
    

def distances_calculations(features_vectors):
    distances=[]
    n=15
    b = [1.0 / n] * n
    a = 1
    for vector1 in tqdm(features_vectors):
        distance=[]
        for vector2 in features_vectors:
            distance.append(scipy.linalg.norm(vector1-vector2,2))
        distances.append(lfilter(b,a,distance))
    return distances
def wasserstein_computation(smooth_data,p1,p2,n_repeats):
    slider = ECGSlider(smooth_data, 6,10).iterator()
    dist = WassersteinDistance(p1)
    dist_dev = WassersteinDistanceDeviation(p2)
    projection_step=1
    curve_transform = SequentialTransformer(
        CurveProjection(
            window=IndicesWindow.range(size=100, step=2),
            step=projection_step
        ),
        PCATransformer(3)
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

def weights_(distances_total:list,alpha:int,beta:int):
    weights=[]
    for dist in tqdm(distances_total):
        w=[]
        for d in (dist):
            w.append(1/(alpha+d))
        weights.append(w)
    weights_new=[]
    for weight in tqdm(weights):
        w=weight
        w=np.array(w)
        w[w<beta]=0
        weights_new.append(w)
    return weights_new

import pickle
def save_pkl(variable, name):
    name = name + '.pkl'
    output = open(name, 'wb')
    pickle.dump(variable, output)
    output.close()

def features_create(model,tensor_set):
    preds=[]
    features=[]
    for curve in tqdm(tensor_set):
      curve=curve.reshape(1,-1)
      inp=Variable(curve)#.cuda()
      feature_vector=model(inp)
      feature_vector=feature_vector.data.cpu().numpy()
      features.append(feature_vector)
    return features