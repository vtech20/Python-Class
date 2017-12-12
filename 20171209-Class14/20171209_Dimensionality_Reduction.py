# -*- coding: utf-8 -*-
"""
Created on Sun Jun 04 15:41:09 2017

@author: karthik.ragunathan
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt


# Step 0: Scale the data if needed (PCA is done covariance matrix which is scale dependent)
# Step 1: Run a correlation analysis
# Step 2: Build PCA model and project the data to new dimensional space
# Step 3: Analyze the variance captured by each pricipal vector
# Step 4: Visualize the data in new dimensional space
# Step 5: Check the factor loadings

irisdata = pd.read_csv("E:\\Python Class\\Data\\iris.csv")

# PCA is an unsupervised dimensionality reduction technique.
# No need of class label
newiris = irisdata.iloc[:,0:4]

# Step 1: Correlation analysis
corr_matrix = newiris.corr()
# Hard to avoid multi-collinearity (correlation of IDVs)
plt.scatter("Petal.Length","Petal.Width",data = newiris) 
plt.scatter("Sepal.Length","Petal.Length",data = newiris)
# only one of the 2 variables are good enough. Can do variable selection

plt.scatter("Sepal.Width","Petal.Length",data = newiris)
# on medium correlated variables, hand picking one of them is hard
# we need a mathematical way of extracting the commonality as one and retaining the 
  # uniqueness individually

# Step 2: PCA model

# Principal Component Analysis projects the data to a different
# dimensional space which are not correlated with each other

# Dim1 = a1*S.L + b1*S.W + c1*P.L + d1*P.W
# Dim2 = a2*S.L + b2*S.W + c2*P.L + d2*P.W
# Dim3 = a3*S.L + b3*S.W + c3*P.L + d3*P.W
# Dim4 = a4*S.L + b4*S.W + c4*P.L + d4*P.W

# First few Principal components would be generally sufficient to explain
    # maximum variance in data and hence facilitating dimensionality reduction

irispca = PCA().fit(newiris)
irispca_projected_matrix = irispca.transform(newiris)

# projecting the raw 4 dimensional data to a new dimensional data using PCA coefficients
iris_projected = pd.DataFrame(irispca.transform(newiris)) 
iris_projected.columns = ["Dim1","Dim2","Dim3","Dim4"] # hard coding of column names
iris_projected.columns = ["Dim" + str(i) for i in range(1,5)] # generating column names in a loop

# Step 3: Variance analysis
irispca.explained_variance_ratio_
np.sum(irispca.explained_variance_ratio_)  # sum of variance explained by all dimensions will be equal to 100%
# screeplot1
plt.bar(np.arange(1,5),irispca.explained_variance_ratio_)
# screeplot2
cumvar = np.cumsum(irispca.explained_variance_ratio_)
plt.bar(np.arange(1,5),cumvar)

# first 2 dimensions are capturing 97.7% variance
 # last 2 dimensions can be ignored
 
#Step 4: Projected Data Visualization

# without class label information
plt.scatter(iris_projected["Dim1"],iris_projected["Dim2"])
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")

# Step 5: factor loadings

factor_loadings = pd.DataFrame(irispca.components_.T)
factor_loadings.columns = ["Dim1","Dim2","Dim3","Dim4"]
factor_loadings.index = newiris.columns

# Dim1 = 0.36*S.L - 0.08*S.W + 0.85*P.L + 0.35*P.W
# Dim2 = 0.65*S.L + 0.73*S.W - 0.17*P.L - 0.07*P.W

# I see 2 groups of data. 
    # One group which is less in Dim 1
    # Another group is significantly high in Dim 1 and marginally high in DIm 2 
    
# Dim 1 is dominated by Petal Length
# Dim 2 is dominated by Sepal Length and Sepal Width

# I see 2 groups of data. 
    # One group which is less in P.L
    # Another group is significantly high in P.L and marginally high in S.L, S.W 
    
########## ASSIGNMENT ON WINE DATA #########################################

winedata = pd.read_csv("E:\\Python Class\\Data\\wine.txt",header = None)
 #Names of 13 attributes can be found in the description. 
 #It is a labelled data set with 3 classes of wines; 
 #note that 1st column in the data corresponds to wine class.
 #Add column names to the data frame. 
 
winedata.shape
winedata.columns = ["Wine_Class", "Alcohol","Malic_acid","Ash","Alcalinity_of_ash",
                       "Magnesium","Total_phenols","Flavanoids","Nonflavanoid_phenols",
                       "Proanthocyanins","Color_intensity","Hue","OD280/OD315",
                       "Proline"]

# Step 0: Scale the data if needed
wine_scaled = pd.DataFrame(scale(winedata.iloc[:,1:14]))
wine_scaled.columns = ["Alcohol","Malic_acid","Ash","Alcalinity_of_ash",
                       "Magnesium","Total_phenols","Flavanoids","Nonflavanoid_phenols",
                       "Proanthocyanins","Color_intensity","Hue","OD280/OD315",
                       "Proline"]
                       
# Step 1: Correlation analysis
corr_matrix = wine_scaled.corr()

####Total Phenols and Flavanoids##############

plt.scatter("Total_phenols","Flavanoids",data = wine_scaled)
plt.xlabel("Total_phenols")
plt.ylabel("Flavanoids")
#Some what good relationship btween Total Phenols and Flavanoids

####OD280/OD315 and Flavanoids################

plt.scatter("OD280/OD315","Flavanoids",data = wine_scaled)
plt.xlabel("OD280/OD315")
plt.ylabel("Flavanoids")
#Some what good relationship btween OD280/OD315 and Flavanoids

####Alcohol and Proline#######################
plt.scatter("Alcohol","Proline",data = wine_scaled)
plt.xlabel("Alcohol")
plt.ylabel("Proline") 
#Somewhat good relationship between alcohol and proline

#####Proanthocyanins and Total phenols###########

plt.scatter("Proanthocyanins","Total_phenols",data = wine_scaled)
plt.xlabel("Proanthocyanins")
plt.ylabel("Total_phenols") 
#Somewhat good relationship Proanthocyanins and Total_phenols

#####OD280/OD315 and Total Phenols#######   
plt.scatter("OD280/OD315","Total_phenols",data = wine_scaled)
plt.xlabel("OD280/OD315")
plt.ylabel("Total_phenols")               
#Somewhat good relationship OD280/OD315 and Total phenols

######
#Ash doesn't have a good relationship with any parameters
#Alcalinity of ash doesn't have a good relationship with any
#Magnesium doesn't have a good relationship with any parameters

#step2
winepca = PCA().fit(wine_scaled)
winepca_projected_matrix = winepca.transform(wine_scaled)
# projecting the raw 4 dimensional data to a new dimensional data using PCA coefficients
wine_projected = pd.DataFrame(winepca_projected_matrix)
wine_projected.columns = ['Dim'+ str(i) for i in range(1,14)] 
                          
# Step 3: Variance analysis
winepca.explained_variance_ratio_    
np.sum(winepca.explained_variance_ratio_)                     

# screeplot1
plt.bar(np.arange(1,14),winepca.explained_variance_ratio_)

# screeplot2
cumvar = np.cumsum(winepca.explained_variance_ratio_)
plt.bar(np.arange(1,14),cumvar)

#when selecting 7 dimentional only we are getting 92% or 6 dimentional when it is 89%

# Step 5: factor loadings

factor_loadings = pd.DataFrame(winepca.components_.T)
factor_loadings.columns = ['Dim'+ str(i) for i in range(1,14)]
factor_loadings.index = wine_scaled.columns