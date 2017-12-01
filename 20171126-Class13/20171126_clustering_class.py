# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 11:24:24 2017

@author: Vignesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

irisdata = pd.read_csv("E:\\Python Class\\data\\iris.csv")
newiris = irisdata.iloc[:,0:4]

#step 1 : Do an Exploratory analysis of variables / Features
#Step 2 : Build Clusters
#Step 3 : Optimise / Evaluate Cluster
#step 4 : Optimize cluster ( choose optional K)
#Step 5 : Vizualize the cluster cetroids
#Step 6 : Cluster Profiling    


#step 1:
plt.scatter("Petal.Length","Petal.Width",data=newiris)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

plt.scatter("Sepal.Length","Sepal.Width",data=newiris)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.scatter("Petal.Length","Sepal.Length",data=newiris)
plt.xlabel("Petal Length")
plt.ylabel("Sepal Length")


plt.scatter("Petal.Width","Sepal.Width",data=newiris)
plt.xlabel("Petal Width")
plt.ylabel("Sepal Width")

newiris.describe()

#Step 2: K Means Clustering
#two groups has been visually seen and let's build 2 cluster model (k=2)
irisclust = KMeans(n_clusters = 2,random_state=100).fit(newiris)
irisclust.labels_
newiris_with_clabel = newiris.copy()    
newiris_with_clabel["Cluster_label"] = irisclust.labels_

plt.scatter("Petal.Length","Petal.Width",c = irisclust.labels_, data=newiris)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

plt.scatter("Sepal.Length","Sepal.Width",c = irisclust.labels_, data=newiris)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.scatter("Petal.Length","Sepal.Length",c = irisclust.labels_, data=newiris)
plt.xlabel("Petal Length")
plt.ylabel("Sepal Length")

plt.scatter("Petal.Width","Sepal.Width",c = irisclust.labels_, data=newiris)
plt.xlabel("Petal Width")
plt.ylabel("Sepal Width")

#Step 3&4: Evaluate clustering and Optimizing Clusters
cluster_centers = irisclust.cluster_centers_

#euclidean distance between each observation and the centroid of the cluster it belongs

cdist_output = np.min(cdist(newiris,irisclust.cluster_centers_,'euclidean'),axis=1)

#Sum of squared distance for K=2

np.sum(cdist_output**2)

#np.min is to find out the closest centroid for that observation
centroid_0 = cluster_centers[0,:]
centroid_1 = cluster_centers[1,:]
newiris_1st_obs = newiris.iloc[0,:].as_matrix()

#distance b/w centroid and 1st observation
((centroid_0[0] - newiris_1st_obs[0])**2 +
 (centroid_0[1] - newiris_1st_obs[1])**2 +
 (centroid_0[2] - newiris_1st_obs[2])**2 +
 (centroid_0[3] - newiris_1st_obs[3])**2)**0.5
 
 
wss = pd.Series([0.0]*10,index = range(1,11)) 
for k in range(1,11):
    irisclust = KMeans(n_clusters = k, random_state=100).fit(newiris)
    cdist_output = np.min(cdist(newiris,irisclust.cluster_centers_,'euclidean'),axis=1)
    wss[k] = np.sum(cdist_output**2)
    
print(wss) 

#Elbow curve
plt.plot(wss)   

#K =2 or 3 is elbow point

#####################Finalize on K=3 ############################################
