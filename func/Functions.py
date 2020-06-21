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
from tqdm import tqdm 
#here additional functions located
dist = WassersteinDistance()

#make std and means equal between 2 distributions
def std_mean_change(was_dev,was):
    new_data=[]
    for i in was_dev:
        new_data.append(np.mean(was)+(i-np.mean(was_dev))*\
                        (np.std(was)/np.std(was_dev)))
    return new_data
#chunk time series to some equal parts
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out 

def recall(df,new_list,sig_series): 
    if   len(new_list)==0 :
        recall=('arrhythmia not detected')
        
    else:
        start_ind=sig_series.index[0]
        last_ind=sig_series.index[-1]
        new_list=(new_list.flatten()).tolist()
        T_p=[]
        Tp_Fn=[]
        for i in (df).tolist():
            if int(i)> int(start_ind) and int(i)<int(last_ind):
                if int(i)  in new_list:
                     T_p.append(i)

                Tp_Fn.append(i)
        if len(Tp_Fn)==0:
            recall=('wrong detection')
        else:
            recall=(len(T_p)/len(Tp_Fn))
        print(len(T_p),len(Tp_Fn))
    return recall

def specifity(df,new_list,sig_series): 
    new_list=(new_list.flatten()).tolist()
    start_ind=sig_series.index[0]
    last_ind=sig_series.index[-1]
    T_n=[]
    F_p=[]
    for i in (df).tolist():
        if int(i)> int(start_ind) and int(i)<int(last_ind):
            if int(i) not in new_list:
                 T_n.append(i)
            if int(i) in new_list:
                F_p.append(i)
    d=len(T_n)+len(F_p)
    
    if d==0:
        spec=('no such data')
    else:
        spec=len(T_n)/d
    return spec,len(T_n),len(F_p)

def precision(df_Norm,df,new_list,sig_series):
    new_list=(new_list.flatten()).tolist()
    start_ind=sig_series.index[0]
    last_ind=sig_series.index[-1]
    T_p=[]
    F_p=[]
    for i in (df_Norm).tolist():
        if int(i)> int(start_ind) and int(i)<int(last_ind):
            if int(i) in new_list:
                F_p.append(i)
    for i in (df).tolist():
        if int(i)> int(start_ind) and int(i)<int(last_ind):
            if int(i) in new_list:
                T_p.append(i)
    d=len(T_p)+len(F_p)
    if d==0:
        prec='no such data'
    else:
        prec=len(T_p)/d
    
    return prec, len(T_p),len(F_p)

def F_score(recall,precision):
    F_score=2*recall*precision/(recall+precision)
    
    return F_score

def accuracy(df_Norm,df,new_list,sig_series):
    new_list=(new_list.flatten()).tolist()
    start_ind=sig_series.index[0]
    last_ind=sig_series.index[-1]
    
    T_p=[]
    F_p=[]
    T_n=[]
    F_n=[]
    for i in (df_Norm).tolist():
        if int(i)> int(start_ind) and int(i)<int(last_ind):
            if int(i) in new_list:
                F_p.append(i)
            else:
                T_n.append(i)
    for i in (df).tolist():
        if int(i)> int(start_ind) and int(i)<int(last_ind):
            if int(i) in new_list:
                T_p.append(i)
            else:
                F_n.append(i)
    d=len(T_p)+len(T_n)+len(F_n)+len(F_p)
    if d==0:
        accur='no such data'
    else:
        accur=(len(T_p)+len(T_n))/d
    return accur

def AUC_score(FPR,TPR):
    AUC=1/2 - FPR/2 + TPR/2
    return AUC

#create new time series only with parts labeled with arrhythmia
def index_to_series(list_ind,data):
    list_ind=(np.array(list_ind).flatten())
    arrhythmia_series=[]
    for i in list_ind:
        for j in i:
            arrhythmia_series.append(data[j])
    return arrhythmia_series  
#for dataset, list of file names
def names(root):
    names=[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            names.append(os.path.join(path, name))
    names=sorted(names)
    return names
#for dataset, list of file names
def data(batch_gen):
    dataiter = iter(batch_gen)
    ecg,name,target_name= dataiter.next()
    ecg=ecg.numpy()
    
    for i in ecg:
        ecg_new=i
    return ecg_new,name,target_name
#calculation of curves(for each window and total)
def curves_calculation(signal,p,n_components,size):
    
    smooth_transform = SequentialTransformer(
    ScaleTransform(0, 1),
    SplineInterpolate(0.01)
    )

    curve_transform = SequentialTransformer(
        CurveProjection(
            window=IndicesWindow.range(size=size, step=5),
            step=1
        ),
        PCATransformer(n_components)
    )

    smooth_data = smooth_transform(signal)
    window_curve=curve_transform(signal)
    dist = WassersteinDistance(p)
    dist_dev = WassersteinDistanceDeviation(p)
    slider = ECGSlider(smooth_data, 6, 200).iterator()
    curves=[]
    total_curve=[]
    for index, window_data in (slider):

        window_curve = curve_transform(window_data)
        curves.append(window_curve)
        total_curve.append(curve_transform(smooth_data[:index[-1]]))
       
    return curves,total_curve
#annotations for files
def annotations(file_number):
    data_f = pd.read_csv('annotations/'+str(file_number)+'annotations.txt')
    r=data_f.iloc[:,0]
    rr=r.str.split(expand=True)
    rr.columns=['time', 'sample','type','sub','chan','Num','Aux']
    data_arrhythmia=rr.loc[(rr['type'] != 'N')&(rr['type'] != '· ') ]
    data_normal=rr.loc[(rr['type'] == 'N')|(rr['type'] == '· ') ]
    return data_arrhythmia,data_normal,rr

#True arrhythmia
def true_labels(data_arrhythmia,sig_series):
    start_ind=sig_series.index[0]
    last_ind=sig_series.index[-1]
    True_labels=[]
    for i in (data_arrhythmia['sample']).tolist():
        if int(i)> int(start_ind) and int(i)<int(last_ind):
             True_labels.append(i)
    return True_labels
def arrhythmia_index(res,sig_series,pad):   
    indexes=[]
    ind=[]
    for k,i in enumerate(res[0].index):
        for j in res[0].index[1:]:
            if (j-i)==1:
               
                indexes.append(np.arange(res[0][i]+sig_series.index[0],res[0][j]+sig_series.index[0]+pad))
                ind.append(i)
                ind.append(j)
    ind=np.unique(ind)
    result =list(set(list(res[0].index))-set(ind.tolist()))
    for i in result:
        indexes.append(np.arange(res[0][i],res[0][i]+pad))
    return indexes
def arrhythmia_index_check(res,sig_series,pad):   
    indexes=np.array([])
    ind=[]
    for k,i in enumerate(res[0].index):
        for j in res[0].index[1:]:
            if (j-i)==1:
               
                indexes=np.append((np.arange(res[0][i]+sig_series.index[0],res[0][j]+sig_series.index[0]+pad)),indexes)
                ind.append(i)
                ind.append(j)
    ind=np.unique(ind)
    result =list(set(list(res[0].index))-set(ind.tolist()))
    for i in result:
        indexes=np.append((np.arange(res[0][i],res[0][i]+pad)),indexes)
    return indexes