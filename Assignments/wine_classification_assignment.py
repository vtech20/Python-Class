# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 08:17:37 2017

@author: Vignesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

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
wine_scaled["Wine_Class"] = winedata.loc[:,"Wine_Class"]                   
wine_scaled["Wine_Class"].value_counts()
sns.lmplot("Total_phenols","Flavanoids",data=wine_scaled,hue="Wine_Class", fit_reg = False)
sns.lmplot("Alcohol","Proline",data=wine_scaled,hue="Wine_Class", fit_reg = False)
sns.lmplot("Proanthocyanins","Total_phenols",data=wine_scaled,hue="Wine_Class", fit_reg = False)

## Step 3: Training Test Split
# Option 2: using inbuilt function in sklearn to do training test split
X_wine_train, X_wine_test, y_wine_train, y_wine_test = \
    train_test_split(wine_scaled.iloc[:,0:13], # IDVs
                     wine_scaled.iloc[:,13], #DVs
                    test_size=0.3, random_state = 30)

# Checking the representation of classes in training and test set
y_wine_train.value_counts()    
y_wine_test.value_counts()

# Step 4: Build Classification model using training data
wine_knn = KNeighborsClassifier(n_neighbors=3).fit(X_wine_train,y_wine_train)

# Step 5: Evaluate the model on test data
predicted_class = wine_knn.predict(X_wine_test)
pd.crosstab(y_wine_test,predicted_class,rownames=["Actual Class"],colnames=["Predicted Class"])
accuracy_score(y_wine_test,predicted_class)
(18+22+13)/54  #98.14%


## CROSS VALIDATION
# Building and evaluating models on different training-test combinations
  # to make sure your model is not working good just one training-test combination
accuracy_all = pd.Series([0.0]*10,index = range(10,101,10))
for seed_i in range(10,101,10):  # looping through different random states   
    X_wine_train, X_wine_test, y_wine_train, y_wine_test = \
    train_test_split(wine_scaled.iloc[:,0:13], 
                     wine_scaled.iloc[:,13], 
                    test_size=0.3, random_state = seed_i)
    wine_knn_model = KNeighborsClassifier(n_neighbors=3).fit(X_wine_train,y_wine_train)
    predicted_wine_class = wine_knn_model.predict(X_wine_test)
    accuracy_all[seed_i] = accuracy_score(y_wine_test,predicted_wine_class)
np.mean(accuracy_all)

#95.9%

accuracy_diff_k = pd.Series([0.0]*10,index = range(1,20,2))
for k_chosen in range(1,20,2): # looping through odd Ks 
    for seed_i in range(10,101,10):    
        X_wine_train, X_wine_test, y_wine_train, y_wine_test = \
        train_test_split(wine_scaled.iloc[:,0:13], 
                         wine_scaled.iloc[:,13], 
                        test_size=0.3, random_state = seed_i)
        wine_knn_model = KNeighborsClassifier(n_neighbors=k_chosen).fit(X_wine_train,y_wine_train)
        predicted_wine_class = wine_knn_model.predict(X_wine_test)
        accuracy_all[seed_i] = accuracy_score(y_wine_test,predicted_wine_class)
    # Average accuracy of all training-test combinations is saved for a given K
    accuracy_diff_k[k_chosen] = np.mean(accuracy_all)
print(accuracy_diff_k) # cross validated accuracy for each K


#1     0.951852
#3     0.959259
#5     0.964815
#7     0.968519
#9     0.966667
#11    0.974074
#13    0.970370
#15    0.970370
#17    0.970370
#19    0.966667
#dtype: float64
#Feels like K=11 is having the good accuracy 97.40%

def wine_cross_validate(model_algo = KNeighborsClassifier()):
    accuracy_all = pd.Series([0.0]*10,index = range(10,101,10))
    for seed_i in range(10,101,10):    
        X_wine_train, X_wine_test, y_wine_train, y_wine_test = \
        train_test_split(wine_scaled.iloc[:,0:13], 
                         wine_scaled.iloc[:,13], 
                        test_size=0.3, random_state = seed_i)
        generic_model = model_algo.fit(X_wine_train,y_wine_train)
        predicted_wine = generic_model.predict(X_wine_test)
        accuracy_all[seed_i] = accuracy_score(y_wine_test,predicted_wine)
    cross_validated_accuracy = np.mean(accuracy_all)
    return (cross_validated_accuracy)

wine_cross_validate(GaussianNB())    #97.22%

wine_cross_validate(DecisionTreeClassifier())  #91.11%

wine_cross_validate(RandomForestClassifier())  #97.03%