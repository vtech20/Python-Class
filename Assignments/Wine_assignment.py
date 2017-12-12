# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 08:18:39 2017

@author: Vignesh
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans #SciKit Learn
from scipy.spatial.distance import cdist
from sklearn.preprocessing import scale


# Step 1: Do an exploratory analysis of Variables/Features
# Step 2: Build clusters (kmeans clustering)
# Step 3: Evaluate cluster
# Step 4: Optimize the cluster (choose an optimal K)
# Step 5: Visualize the clusters with centroids
# Step 6: Cluster Profiling
cols1 = ['Class','Alcohol','Malic acid','Ash','Alcalinity of ash',
       'Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
       'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
       'Proline']
winedata = pd.read_csv("E:\\Python Class\\Data\\wine.txt",
                        names = cols1 )
#Reading the file from web itself and naming the column
ss = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",names=cols1)

newwinedata = winedata.iloc[:,1:14]

# Step 1: Do an exploratory analysis of Variables/Features
newwinedata.describe()

plt.scatter("Alcohol","Malic acid",data = newwinedata)
plt.xlabel("Alcohol")
plt.ylabel("Malic acid")
#Not a good relationship as well as difficult to cluster
plt.scatter("Alcohol","Ash",data = newwinedata)
plt.xlabel("Alcohol")
plt.ylabel("Ash")
#somehow we can say that when alcohol is increasing Ash also increases but still
#difficult to cluster
plt.scatter("Alcohol","Magnesium",data = newwinedata)
plt.xlabel("Alcohol")
plt.ylabel("Magnesium")
#somehow we can say that when alcohol is increasing Magnesium also increases but still
#difficult to cluster
plt.scatter("Ash","Alcalinity of ash",data = newwinedata)
plt.xlabel("Ash")
plt.ylabel("Alcalinity of ash")
#somehow we can say that when Ash is increasing Alcalinity of ash also increases but still
#difficult to cluster
plt.scatter("Alcohol","Proline",data = newwinedata)
plt.xlabel("Alcohol")
plt.ylabel("Proline")

plt.scatter("Flavanoids","Proline",data = newwinedata)
plt.xlabel("Flavanoids")
plt.ylabel("Proline")

plt.scatter("Color intensity","Flavanoids",data = newwinedata)
plt.xlabel("Color intensity")
plt.ylabel("Flavanoids")

plt.scatter("Flavanoids","Hue",data = newwinedata)
plt.xlabel("Flavanoids")
plt.ylabel("Hue")

plt.scatter("Alcohol","Color intensity",data = newwinedata)
plt.xlabel("Alcohol")
plt.ylabel("Color intensity")

#somehow we can cluster but still difficult
axes = pd.tools.plotting.scatter_matrix(newwinedata, alpha=0.5)
corr_matrix = newwinedata.corr()

## Step 2: K Means Clustering
# somehow 3 groups were visually seen. Let's build 3 cluster model (k = 3)
#Run k means clustering on the attributes with k = 3. 
wine_clust = KMeans(n_clusters = 3,random_state=100).fit(newwinedata)
wine_clust.labels_
newwine_with_lbl = newwinedata.copy()
newwine_with_lbl["clustlabel"] = wine_clust.labels_

#visualize
plt.scatter("Alcohol","Proline",c = wine_clust.labels_,data = newwinedata)
plt.xlabel("Alcohol")
plt.ylabel("Proline")

plt.scatter("Color intensity","Proline",c = wine_clust.labels_,data = newwinedata)
plt.xlabel("Color intensity")
plt.ylabel("Proline")

plt.scatter("Ash","Alcalinity of ash",c = wine_clust.labels_,data = newwinedata)
plt.xlabel("Ash")
plt.ylabel("Alcalinity of ash")

plt.scatter("OD280/OD315 of diluted wines","Flavanoids",c=wine_clust.labels_,data = newwinedata)
plt.xlabel("OD280/OD315 of diluted wines")
plt.ylabel("Flavanoids")

plt.scatter("Flavanoids","Proline",c=wine_clust.labels_,data = newwinedata)
plt.xlabel("Flavanoids")
plt.ylabel("Proline")
cluster_centers = wine_clust.cluster_centers_
cdist_output = np.min(cdist(newwinedata,wine_clust.cluster_centers_,'euclidean'),axis=1)
#What would have been your suggestion for number of clusters (k) if it was not provided as input? 

wss = pd.Series([0.0]*10,index = range(1,11))
for k in range(1,11):
    wine_clust = KMeans(n_clusters = k,random_state=100).fit(newwinedata)
    cdist_output = np.min(cdist(newwinedata,wine_clust.cluster_centers_,'euclidean'),axis=1)
    wss[k] = np.sum(cdist_output**2)

print(wss)

# Elbow curve
plt.plot(wss)

#K could be 2 or 3 or 4 wheras 4 is the maximum
# euclidean distance between each observation and the centroid of the cluster it belongs to
within_cluster_distance = np.sum(cdist_output**2)
#218020.48

wine_clust_2 = KMeans(n_clusters = 2,random_state=100).fit(newwinedata)
wine_clust_2.labels_
newwine_with_lbl2 = newwinedata.copy()
newwine_with_lbl2["clustlabel"] = wine_clust_2.labels_

#Rerun the clustering with just Proline attribute (last column in data).
newwine_proline = newwinedata.loc[:,['Alcohol','Proline']]
newwine_proline_new = pd.DataFrame(newwinedata.loc[:,'Proline'])
wine_clust_pro = KMeans(n_clusters = 2,random_state=100).fit(newwine_proline_new)
wine_clust_pro.labels_
cdist_output_1 = np.min(cdist(newwine_proline_new,wine_clust_pro.cluster_centers_,'euclidean'),axis=1)
within_cluster_distance_1 = np.sum(cdist_output_1**2)
#4507884.22
#Comparing the cluster labels of 2 cluster systems
#using all 13 attribute

pd.crosstab(wine_clust_2.labels_,wine_clust_pro.labels_)

#Scale of proline is significantly high which is biasing the clusturing output
#If variables are of different measure units, they have to be scalled

# (x - Mu)/ Sig
newwine_scale = pd.DataFrame(scale(newwinedata))
newwine_scale.columns= ['Alcohol','Malic acid','Ash','Alcalinity of ash',
       'Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
       'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
       'Proline']
       
newwine_scale.describe()

# Function which will iterate through different Ks and plot the elbow curves

def choose_K(df):    
    wss = pd.Series([0.0]*10,index = range(1,11))
    for k in range(1,11):
        dfclust = KMeans(n_clusters = k, random_state=100).fit(df)
        dist = np.min(cdist(df, 
                dfclust.cluster_centers_, 'euclidean'),axis=1)
        wss[k] = np.sum(dist**2)
    plt.plot(wss)

choose_K(newwine_scale)
