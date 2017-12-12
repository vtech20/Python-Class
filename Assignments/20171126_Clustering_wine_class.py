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

# Step 1: Do an exploratory analysis of Variables/Features
# Step 2: Build clusters (kmeans clustering)
# Step 3: Evaluate cluster
# Step 4: Optimize the cluster (choose an optimal K)
# Step 5: Visualize the clusters with centroids
# Step 6: Cluster Profiling

# data with measurements of features in iris flower species
irisdata_with_label = pd.read_csv("data/iris.csv")

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

# The 3 groups are well separated by Petal Length
# Quite some overlap between 2 groups on Petal Width
# Quite some overlaps between groups in Sepal Length 
# Completely overlapping clusters on Sepal Width

# Function which will iterate through different Ks and plot the elbow curves

def choose_K(df):    
    wss = pd.Series([0.0]*10,index = range(1,11))
    for k in range(1,11):
        dfclust = KMeans(n_clusters = k, random_state=100).fit(df)
        dist = np.min(cdist(df, 
                dfclust.cluster_centers_, 'euclidean'),axis=1)
        wss[k] = np.sum(dist**2)
    plt.plot(wss)

choose_K(newiris) 

#################### Wine Data set #################################3
# 1. Download Wine dataset (wine.data) from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/wine).
# Note: This website is a good resource for open data


# 2. Read the wine.data file as data frame (name the dataframe as winedata). 
# Don't be concerned that it is not a .csv file. 
# Open the document in text editor and see that it indeed has comma separated values. 
# You can read this file using regular csv reading functions. 
# Note that the file does not have header (column headings)
winedata = pd.read_csv("data/wine.data",header = None)
# Additional Note: You could directly read the data from the web by 
# passing url as the path argument for csv reading function

winedata = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header = None)


# 3. Names of 13 attributes can be found in the description. 
# It is a labelled data set with 3 classes of wines; 
# note that 1st column in the data corresponds to wine class and has to be removed. 
# Add column names to the data frame.


winedata.columns = ["Wine_Class", "Alcohol","Malic_acid","Ash","Alcalinity_of_ash",
                       "Magnesium","Total_phenols","Flavanoids","Nonflavanoid_phenols",
                       "Proanthocyanins","Color_intensity","Hue","OD280/OD315",
                       "Proline"]

# 4. Do a descriptive analytics by comparing mean, median etc. of 13 attributes 
newwine = winedata.iloc[:,1:14]
newwine.describe()

#5. Create scatter plots for some attribute combinations (trying out for all combinations
 #would be exhaustive – dimensionality is a curse!).
 
 # Flavanoids, Malic Acid, Proline, Color Intensity
plt.scatter(newwine["Flavanoids"],newwine["Proline"])
plt.scatter(newwine["Color_intensity"],newwine["Malic_acid"])
plt.scatter(newwine["Color_intensity"],newwine["Flavanoids"])

# 6. Run k means clustering on the attributes with k = 3.
wineclust = KMeans(n_clusters = 3).fit(newwine)

newwine2 = newwine.copy()
newwine2["cluster_no"] = wineclust.labels_
wine_centroids = wineclust.cluster_centers_

# 7. Visualize scatter plots generated in Step 6 with colors from cluster numbers
plt.scatter(newwine["Flavanoids"],newwine["Proline"],c = wineclust.labels_)
plt.scatter(newwine["Color_intensity"],newwine["Flavanoids"], c = wineclust.labels_)
plt.scatter(newwine["Flavanoids"],newwine["Malic_acid"],c = wineclust.labels_)
plt.scatter(newwine["Color_intensity"],newwine["Proline"], c = wineclust.labels_)
plt.scatter(newwine["Alcohol"],newwine["Proline"], c = wineclust.labels_)


# 8. What would have been your suggestion for number of clusters (k) 
# if it was not provided as input?
choose_K(newwine)

winecluster = KMeans(n_clusters = 2,random_state = 100).fit(newwine)
newwine2["cluster_no"] = winecluster.labels_
plt.scatter(newwine["Flavanoids"],newwine["Proline"],c = winecluster.labels_)

plt.scatter(newwine["Color_intensity"],newwine["Flavanoids"], c = winecluster.labels_)

#9. Rerun the clustering with just Proline attribute 
# (last column in data). 
proilinedf = pd.DataFrame(newwine["Proline"])
prolinecluster = KMeans(n_clusters = 2,random_state = 100).fit(proilinedf)
newwine2["cluster_no_proline"] = prolinecluster.labels_

"""
Compare the cluster performance of just Proline 
attribute (Q 9) vs All attributes (Q 6). 
If they are comparable, what do you think is 
the value addition from other 12 attributes? 
What is special about the values in Proline column?
"""
# Cross tab for comparing results of 2 decision systems
a1 = np.array([0,0,1,1,0])
a2 = np.array([0,0,1,1,1])
a3 = np.array([1,1,0,0,1])
pd.crosstab(a1,a2)
pd.crosstab(a1,a3)

pd.crosstab(winecluster.labels_,prolinecluster.labels_)
