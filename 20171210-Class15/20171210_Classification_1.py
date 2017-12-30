# -*- coding: utf-8 -*-

# Regression and CLassification are Supervised Learning techniques  (has DV)
 # In Regression DV is continuous number (e.g: Sales volume, Heart Weight, Availability)
 # In classification DV is a categorical variable (e.g: Good or Bad risk, Spam or not)
# Clustering and Dimensionality Reduction are Unsupervised learning techniques  (no DV)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Step 0: Scale the variables for distance based algorithms like KNN
# Step 1: Identify IDV and DV (variable to be classified)
# Step 2: Do a descriptive statistics and visualize
# Step 3: Divide the data to Training and Test
# Step 4: Build classification model on training data
# Step 5: Evaluate the model using test data (Confusion matrix, accuracy etc.)
      # Evaluate model for different training-test combination (Cross Validation)
# Step 6: Model fine tuning if needed
# Step 7: Compare modeling approaches and decide a model for future predictions


irisdata = pd.read_csv("E:\\Python Class\\Data\\iris.csv")

## Step 1:
# DV: Species
# IDV: S.L, S.W, P.L, P.W

## Step 2: Descriptive statistics
print(irisdata.groupby("Species").agg(np.mean))
irisdata.boxplot(column=["Sepal.Length"],by = ["Species"])
irisdata.boxplot(column=["Petal.Width"],by = ["Species"])
irisdata.boxplot(column=["Sepal.Width"],by = ["Species"])
irisdata.boxplot(column=["Sepal.Length","Petal.Length"],by = ["Species"])

plt.scatter("Petal.Length","Sepal.Length",data = irisdata)

sns.lmplot("Sepal.Length","Sepal.Width",data = irisdata, hue = "Species", fit_reg = False)
sns.lmplot("Petal.Length","Petal.Width",data = irisdata, hue = "Species", fit_reg = False)


## Step 3: Training Test Split
# Option 1: The Hard Way
all_rows = np.arange(irisdata.shape[0])
np.random.seed(10)
training_samples = np.random.choice(all_rows,int(0.7*irisdata.shape[0]),replace = False)
test_samples = all_rows[~np.in1d(all_rows,training_samples)]
training_data = irisdata.iloc[training_samples,:]
test_data = irisdata.iloc[test_samples,:]
X_iris_train = training_data.iloc[:,0:4] # IDVs of training data
X_iris_test = test_data.iloc[:,0:4] # IDVs of test data
y_iris_train = training_data.iloc[:,4] # DV of training data
y_iris_test = test_data.iloc[:,4] # DV of test data

# Option 2: using inbuilt function in sklearn to do training test split
X_iris_train, X_iris_test, y_iris_train, y_iris_test = \
    train_test_split(irisdata.iloc[:,0:4], # IDVs
                     irisdata.iloc[:,4], #DVs
                    test_size=0.3, random_state = 30)
    
# Checking the representation of classes in training and test set
y_iris_train.value_counts()
y_iris_test.value_counts()


# Step 4: Build Classification model using training data
iris_knn = KNeighborsClassifier(n_neighbors=3).fit(X_iris_train,y_iris_train)

# Step 5: Evaluate the model on test data
predicted_species_knn = iris_knn.predict(X_iris_test)
pd.crosstab(y_iris_test,predicted_species_knn,
            rownames = ["Actual Species"], 
            colnames = ["Predicted Species"])
(13 + 11 + 18)/45 # 93.33% accuracy

## CROSS VALIDATION
# Building and evaluating models on different training-test combinations
  # to make sure your model is not working good just one training-test combination
accuracy_all = pd.Series([0.0]*10,index = range(10,101,10))
for seed_i in range(10,101,10):  # looping through different random states   
    X_iris_train, X_iris_test, y_iris_train, y_iris_test = \
    train_test_split(irisdata.iloc[:,0:4], 
                     irisdata.iloc[:,4], 
                    test_size=0.3, random_state = seed_i)
    iris_knn_model = KNeighborsClassifier(n_neighbors=3).fit(X_iris_train,y_iris_train)
    predicted_species_knn = iris_knn_model.predict(X_iris_test)
    accuracy_all[seed_i] = accuracy_score(y_iris_test,predicted_species_knn)
np.mean(accuracy_all)

## choosing different K and doing cross validation for each K
accuracy_diff_k = pd.Series([0.0]*10,index = range(1,20,2))
for k_chosen in range(1,20,2): # looping through odd Ks 
    for seed_i in range(10,101,10):    
        X_iris_train, X_iris_test, y_iris_train, y_iris_test = \
        train_test_split(irisdata.iloc[:,0:4], 
                         irisdata.iloc[:,4], 
                        test_size=0.3, random_state = seed_i)
        iris_knn_model = KNeighborsClassifier(n_neighbors=k_chosen).fit(X_iris_train,y_iris_train)
        predicted_species_knn = iris_knn_model.predict(X_iris_test)
        accuracy_all[seed_i] = accuracy_score(y_iris_test,predicted_species_knn)
    # Average accuracy of all training-test combinations is saved for a given K
    accuracy_diff_k[k_chosen] = np.mean(accuracy_all)
print(accuracy_diff_k) # cross validated accuracy for each K

# K = 7 has a good accuracy and can be chosen as the best model
# It is also found that the classification accuracy is decent irrespective of K

## Function which does cross validation for any algo
def iris_cross_validate(model_algo = KNeighborsClassifier()): # Functional Programming
    accuracy_all = pd.Series([0.0]*10,index = range(10,101,10))
    for seed_i in range(10,101,10):    
        X_iris_train, X_iris_test, y_iris_train, y_iris_test = \
        train_test_split(irisdata.iloc[:,0:4], 
                         irisdata.iloc[:,4], 
                        test_size=0.3, random_state = seed_i)
        generic_model = model_algo.fit(X_iris_train,y_iris_train) # fitting any model which comes as input
        predicted_species = generic_model.predict(X_iris_test)
        accuracy_all[seed_i] = accuracy_score(y_iris_test,predicted_species)
    cross_validated_accuracy = np.mean(accuracy_all)
    return (cross_validated_accuracy)

iris_cross_validate(KNeighborsClassifier(n_neighbors = 3)) #96.44
iris_cross_validate(KNeighborsClassifier(n_neighbors = 5)) #96.66



