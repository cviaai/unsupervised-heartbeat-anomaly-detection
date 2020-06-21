import numpy as np
import infomap
import pandas as pd
import scipy
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.signal import lfilter
import warnings

from tqdm import tqdm_notebook
from functools import reduce
warnings.filterwarnings('ignore')

def cluster_counts(labels,features):
    cluster_indexes=[]
    for c in np.unique(labels):
        cluster_indexes.append(np.where(np.array(labels)==c))
    clusters_features=[]
    for cluster in cluster_indexes:
      # print(cluster)
      cluster_features=[features[j] for j in cluster[0] ]
      clusters_features.append(cluster_features)
    return cluster_indexes,clusters_features

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
def weights_init(distances_total):
  weights=[]
  for dist in tqdm(distances_total):
      w=[]
      for d in (dist):
          w.append(1/(0.01+d))
      weights.append(w)
  return weights

import pickle
def save_pkl(variable, name):
    name = name + '.pkl'
    output = open(name, 'wb')
    pickle.dump(variable, output)
    output.close()
def findCommunities(G):
    """
    Partition network with the Infomap algorithm.
    Annotates nodes with 'community' id.
    """

    im = infomap.Infomap("--two-level")

    print("Building Infomap network from a NetworkX graph...")
    for source, target in G.edges:
        im.add_link(source, target)

    print("Find communities with Infomap...")
    im.run()

    print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")

    communities = im.get_modules()
    nx.set_node_attributes(G, communities, 'community')


#create function for  clearence inside the cluster, by euclidian distance
def new_weights(weights):
  weights_new=[]
  for weight in weights:
    w=weight.copy()
    w=np.array(w)
    w[w<(np.median(w)+np.std(w))]=0
    weights_new.append(w)
  return weights_new

def cluster_clearence(cluster_feature_vector:list):
    print(len(cluster_feature_vector))
    distance=[]
    for vector_1 in tqdm_notebook(cluster_feature_vector):
        dist=[]
        for vector_2 in cluster_feature_vector:
            dist.append(np.linalg.norm(vector_1-vector_2,2))
        dist=np.mean(dist)
        distance.append(dist)
    return np.mean(distance)

def cluster_creation(distances:list,features_all:list,features:list,inter:int,treshold:float):
  final_clusters=[]

  weights=weights_init(distances)
  weights_new=new_weights(weights)

  weights_matrix=np.matrix(weights_new)
  Graph=nx.DiGraph(weights_matrix)
  findCommunities(Graph)
  communities = [v for k,v in nx.get_node_attributes(Graph, 'community').items()]
 # print('number of clusters:', np.unique(communities))
  clusters,cluster_features=cluster_counts(communities,features_all)

  # clusters=[cluster[0] for cluster in clusters ]
  # vectors=[distances for i in range(len(clusters))]
  distance=list(map(cluster_clearence,cluster_features))
  if np.isnan(np.array(distance)).any() == True:
    print('DETECTED NAN')
    a=np.argwhere(np.isnan(distance))
   # print(a)
    distance.pop(a[0][0])
    cluster_features.pop(a[0][0])

  print('clearence of clusters',distance)
  
  bad_clusters=[]
  for i in range(len(distance)):
    
    if distance[i]<=treshold:
      final_clusters.append(cluster_features[i])
    else:
      bad_clusters.append(cluster_features[i])
  cluster_length=len(bad_clusters)

  i=0
  # for i in range(len(bad_clusters)):
  final_bad_clusters=[]
  while i<cluster_length:
    #print('ITERATION '+str(i),'Total quantaty of clusters:',len(bad_clusters))
    #print("___"*20)
    small_features=bad_clusters[i]
    small_vectors=distances_calculations(small_features)

    weights=weights_init(small_vectors)
    weights_new=new_weights(weights)
    
    weights_matrix=np.matrix(weights_new)
    
    Graph=nx.DiGraph(weights_matrix)
    findCommunities(Graph)
    communities = [v for k,v in nx.get_node_attributes(Graph, 'community').items()]
    #print('number of mini clusters in cluster '+str(i), np.unique(communities))

    clusters,cluster_features=cluster_counts(communities,small_features)

    # clusters=[cluster[0] for cluster in clusters ]
    # vectors=[small_vectors for vec in range(len(clusters))]
    distance=list(map(cluster_clearence,cluster_features))
    if np.isnan(np.array(distance)).any() == True:
      print('DETECTED NAN')
      a=np.argwhere(np.isnan(distance))
      distance.pop(a[0][0])
      cluster_features.pop(a[0][0])
    print('clearence of mini clusters',distance)

    for m in range(len(distance)):
      if distance[m]<=treshold:
        final_clusters.append(cluster_features[m])
      elif i<=inter and distance[m]>treshold:
        bad_clusters.append(cluster_features[m])

      elif i>inter and distance[m]>treshold:
        final_bad_clusters.append(cluster_features[m])
      
    cluster_length=len(bad_clusters)
    print(cluster_length)
    i=i+1
   
  bad_clusters_vectors=np.concatenate(final_bad_clusters)
  bad_vectors=distances_calculations(bad_clusters_vectors)

  weights=weights_init(bad_vectors)
  weights_new=new_weights(weights)

  weights_matrix=np.matrix(weights_new)
  Graph=nx.DiGraph(weights_matrix)
  findCommunities(Graph)
  communities = [v for k,v in nx.get_node_attributes(Graph, 'community').items()]

 # print('number of mini clusters in cluster '+str(i), np.unique(communities))

  clusters,cluster_features=cluster_counts(communities,bad_clusters_vectors)

  
  bad_clusters=[]
  for i in range(len(distance)):
    
    if distance[i]<=treshold:
      final_clusters.append(cluster_features[i])
    else:
      bad_clusters.append(cluster_features[i])
  cluster_length=len(bad_clusters)
  i=0
 # print('CLUSTERS from bad')
  final_bad_clusters=[]
  while i<cluster_length:
   # print('ITERATION '+str(i),'Total quantaty of clusters from bad cluster:',len(bad_clusters))
    #print("___"*20)
    small_features=bad_clusters[i]
    small_vectors=distances_calculations(small_features)
    
    weights=weights_init(small_vectors)
    weights_new=new_weights(weights)

    weights_matrix=np.matrix(weights_new)
    Graph=nx.DiGraph(weights_matrix)
    findCommunities(Graph)
    communities = [v for k,v in nx.get_node_attributes(Graph, 'community').items()]
    #print('number of mini clusters in cluster '+str(i), np.unique(communities))

    clusters,cluster_features=cluster_counts(communities,small_features)

    # clusters=[cluster[0] for cluster in clusters ]
    # vectors=[small_vectors for vec in range(len(clusters))]
    distance=list(map(cluster_clearence,cluster_features))
    if np.isnan(np.array(distance)).any() == True:
      #print('DETECTED NAN')
      a=np.argwhere(np.isnan(distance))
      distance.pop(a[0][0])
      cluster_features.pop(a[0][0])
    #print('clearence of mini clusters from bad clusters',distance)

    for m in range(len(distance)):
      if distance[m]<=treshold:
        final_clusters.append(cluster_features[m])
      elif i<=inter and distance[m]>treshold:
        bad_clusters.append(cluster_features[m])

      elif i>inter and distance[m]>treshold:
        final_bad_clusters.append(cluster_features[m])
      
    cluster_length=len(bad_clusters)
    # print(cluster_length)
    i=i+1
 
  return final_clusters,final_bad_clusters

def final_cluster(features,cluster_feature):
  final_clusters=[]
  for feature_vector in cluster_feature:
    cluster=[]
    for vector in feature_vector:
      bc=np.bincount(np.where(features==vector)[0])
      cluster.append(bc.argmax())
    final_clusters.append(cluster)
  return final_clusters