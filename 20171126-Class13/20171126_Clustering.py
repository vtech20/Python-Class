# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 11:25:25 2017

@author: karthik.ragunathan
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans #SciKit Learn
from scipy.spatial.distance import cdist
import os
# Step 1: Do an exploratory analysis of Variables/Features
# Step 2: Build clusters (kmeans clustering)
# Step 3: Evaluate cluster
# Step 4: Optimize the cluster (choose an optimal K)
# Step 5: Visualize the clusters with centroids
# Step 6: Cluster Profiling

# data with measurements of features in iris flower species
os.chdir('E:\Python Class\Data')
irisdata_with_label = pd.read_csv("iris.csv")

## Step 1: Exploratory Analysis
newiris = irisdata_with_label.iloc[:,0:4]

avg_sl = np.mean(newiris["Sepal.Length"])
group1 = newiris.loc[newiris["Sepal.Length"] > avg_sl,:]
group2 = newiris.loc[newiris["Sepal.Length"] <= avg_sl,:]

newiris["Petal.Length"].hist()
newiris["Petal.Width"].hist()
newiris["Sepal.Width"].hist()
newiris["Sepal.Length"].hist()

plt.scatter("Petal.Length","Petal.Width",data = newiris)
plt.xlabel("Petal.Length")
plt.ylabel("Petal.Width")

plt.scatter("Sepal.Length","Sepal.Width",data = newiris)
plt.xlabel("Sepal.Length")
plt.ylabel("Sepal.Width")

plt.scatter("Petal.Length","Sepal.Length",data = newiris)
plt.xlabel("Petal.Length")
plt.ylabel("Sepal.Length")

plt.scatter("Petal.Width","Sepal.Width",data = newiris)
plt.xlabel("Petal.Width")
plt.ylabel("Sepal.Width")

newiris.describe()

## Step 2: K Means Clustering
# Two groups were visually seen. Let's build 2 cluster model (k = 2)

irisclust = KMeans(n_clusters = 2,random_state=100).fit(newiris)
irisclust.labels_ # label 0 and label 1
newiris_with_clulabel = newiris.copy()
newiris_with_clulabel["Cluster_Label"] = irisclust.labels_

plt.scatter("Petal.Length","Petal.Width",c =irisclust.labels_, data = newiris)
plt.xlabel("Petal.Length")
plt.ylabel("Petal.Width")

plt.scatter("Sepal.Length","Sepal.Width",c =irisclust.labels_,data = newiris)
plt.xlabel("Sepal.Length")
plt.ylabel("Sepal.Width")

plt.scatter("Petal.Length","Sepal.Length",c =irisclust.labels_,data = newiris)
plt.xlabel("Petal.Length")
plt.ylabel("Sepal.Length")

plt.scatter("Petal.Width","Sepal.Width",c =irisclust.labels_,data = newiris)
plt.xlabel("Petal.Width")
plt.ylabel("Sepal.Width")

## Step 3&4: Evaluate Clustering and Optimizing Clusters 
cluster_centers = irisclust.cluster_centers_

# euclidean distance between each observation and the centroid of the cluster it belongs to
cdist_output = np.min(cdist(newiris,irisclust.cluster_centers_,'euclidean'),axis=1)

# Sum of squared distance for K = 2
within_cluster_distance = np.sum(cdist_output**2)

centroid_o = cluster_centers[0,:]
centroid_1 = cluster_centers[1,:]
newiris_1st_observation = newiris.iloc[0,:].as_matrix()
newiris_2nd_observation = newiris.iloc[1,:].as_matrix()
newiris_last_observation = newiris.iloc[149,:].as_matrix()
#####3 hand calculation of euclidean distance #####################
# euclidean distance between centroid_o and 1st observation
((centroid_o[0] - newiris_1st_observation[0])**2 + 
(centroid_o[1] - newiris_1st_observation[1])**2 +
(centroid_o[2] - newiris_1st_observation[2])**2 + 
(centroid_o[3] - newiris_1st_observation[3])**2)**0.5
  
# euclidean distance between centroid_o and 2nd observation
((centroid_o[0] - newiris_2nd_observation[0])**2 + 
(centroid_o[1] - newiris_2nd_observation[1])**2 +
(centroid_o[2] - newiris_2nd_observation[2])**2 + 
(centroid_o[3] - newiris_2nd_observation[3])**2)**0.5

# euclidean distance between centroid_1 and last observation
((centroid_1[0] - newiris_last_observation[0])**2 + 
(centroid_1[1] - newiris_last_observation[1])**2 +
(centroid_1[2] - newiris_last_observation[2])**2 + 
(centroid_1[3] - newiris_last_observation[3])**2)**0.5
############################################################################
#from scipy.spatial.distance import cdist

  # Building clusters with k=1 till k=10 and comparing the
  #within cluster distance between each clustering output
wss = pd.Series([0.0]*10,index = range(1,11))
for k in range(1,11):
    irisclust = KMeans(n_clusters = k,random_state=100).fit(newiris)
    cdist_output = np.min(cdist(newiris,irisclust.cluster_centers_,'euclidean'),axis=1)
    wss[k] = np.sum(cdist_output**2)

print(wss)

# Elbow curve
plt.plot(wss)

# K= 2 or 3 is the elbow point

##### Finalize on K = 3 ##############################3
irisclust = KMeans(n_clusters = 3,random_state=100).fit(newiris)
irisclust.labels_ # label 0 and label 1
newiris_with_clulabel = newiris.copy()
newiris_with_clulabel["Cluster_Label"] = irisclust.labels_

## Step 5
plt.scatter("Petal.Length","Petal.Width",c =irisclust.labels_, data = newiris)
plt.scatter(irisclust.cluster_centers_[:,2],irisclust.cluster_centers_[:,3],
            c = np.unique(irisclust.labels_),marker = 's',s = 200)
plt.xlabel("Petal.Length")
plt.ylabel("Petal.Width")

plt.scatter("Sepal.Length","Sepal.Width",c =irisclust.labels_,data = newiris)
plt.scatter(irisclust.cluster_centers_[:,0],irisclust.cluster_centers_[:,1],
            c = np.unique(irisclust.labels_),marker = 's',s = 200)
plt.xlabel("Sepal.Length")
plt.ylabel("Sepal.Width")

# Step 6: Cluster Profiling
newiris_with_clulabel.groupby("Cluster_Label").agg(np.mean)
