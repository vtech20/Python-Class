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
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import scale

# Step 0: Scale the variables for distance based algorithms like KNN
# Step 1: Identify IDV and DV (variable to be classified)
# Step 2: Do a descriptive statistics and visualize
# Step 3: Divide the data to Training and Test
# Step 4: Build classification model on training data
# Step 5: Evaluate the model using test data (Confusion matrix, accuracy etc.)
      # Evaluate model for different training-test combination (Cross Validation)
# Step 6: Model fine tuning if needed
# Step 7: Compare modeling approaches and decide a model for future predictions


irisdata = pd.read_csv("data/iris.csv")

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
#all_rows = np.arange(irisdata.shape[0])
#np.random.seed(10)
#training_samples = np.random.choice(all_rows,int(0.7*irisdata.shape[0]),replace = False)
#test_samples = all_rows[~np.in1d(all_rows,training_samples)]
#training_data = irisdata.iloc[training_samples,:]
#test_data = irisdata.iloc[test_samples,:]
#X_iris_train = training_data.iloc[:,0:4] # IDVs of training data
#X_iris_test = test_data.iloc[:,0:4] # IDVs of test data
#y_iris_train = training_data.iloc[:,4] # DV of training data
#y_iris_test = test_data.iloc[:,4] # DV of test data

# Option 2: using inbuilt function in sklearn to do training test split
X_iris_train, X_iris_test, y_iris_train, y_iris_test = \
    train_test_split(irisdata.iloc[:,0:4], # IDVs
                     irisdata.iloc[:,4], #DVs
                    train_size=0.7, random_state = 30)
    
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

accuracy_diff_k = pd.Series([0.0]*10,index = range(1,20,2))
for k_chosen in range(1,20,2): # looping through odd Ks 
    accuracy_diff_k[k_chosen] = \
       iris_cross_validate(KNeighborsClassifier(n_neighbors = k_chosen))
print (accuracy_diff_k)

# Gaussian Naive Bayes
iris_cross_validate(GaussianNB()) # 95.77

# Decision Tree
iris_cross_validate(DecisionTreeClassifier()) #95.77

# Random Forest
iris_cross_validate(RandomForestClassifier()) #94.88

############## Wine Data set  ###############################################
winedata = pd.read_csv("data/wine.data",header = None)
 #Names of 13 attributes can be found in the description. 
 #It is a labelled data set with 3 classes of wines; 
 #note that 1st column in the data corresponds to wine class.
 #Add column names to the data frame. 
 
winedata.shape
winedata.columns = ["Wine_Class", "Alcohol","Malic_acid","Ash","Alcalinity_of_ash",
                       "Magnesium","Total_phenols","Flavanoids","Nonflavanoid_phenols",
                       "Proanthocyanins","Color_intensity","Hue","OD280/OD315",
                       "Proline"]

# DV: Wine Class
# IDV: 13 attributes

# Step 0: Scale the features
winefeatures = winedata.iloc[:,1:14]
winefeatures_scaled = pd.DataFrame(scale(winefeatures))
winefeatures_scaled.columns = ["Alcohol","Malic acid", "Ash","Alcalinity of ash",
                     "Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols",
                     "Proanthocyanins","Color intensity","Hue",
                     "OD280/OD315 of diluted wines",
                     "Proline"]
winefeatures_scaled["class"] = winedata["Wine_Class"]
# Step 1: DV and IDV selection
# DV: Class (3 wine classes)
# IDV: Alcohol,....., Proline

# Step 2: Do descriptive statistics
winefeatures_scaled.groupby("class").agg(np.mean)
sns.lmplot("Alcohol","Flavanoids",data = winefeatures_scaled, 
           hue = "class",fit_reg = False)

# Step 3: Divide the data to training and test set (70-30)
X_wine_train, X_wine_test, y_wine_train, y_wine_test = \
    train_test_split(winefeatures_scaled.iloc[:,0:14], 
                     winefeatures_scaled.loc[:,"class"], 
                    test_size=0.3, random_state = 30)
# Checking the representation of classes in training and test set
y_wine_train.value_counts()
y_wine_test.value_counts()

# Step 4: Build Classification model (knn)
wine_knn3 = KNeighborsClassifier(n_neighbors=3).fit(X_wine_train,y_wine_train)
predicted_wine_knn3 = wine_knn3.predict(X_wine_test)

# Step 5: Evaluate the model
pd.crosstab(y_wine_test,predicted_wine_knn3,
            rownames = ["Actual Wine Class"], 
            colnames = ["Predicted Wine Class"])
accuracy = (18+22+13)/54
print (accuracy) # 98.14
# using inbuilt function for accuracy prediction
accuracy_score(y_wine_test,predicted_wine_knn3)

# Generic function for cross validating any classification algorithm
def wine_cross_validate(model_algo = KNeighborsClassifier()):
    accuracy_all = pd.Series([0.0]*10,index = range(10,101,10))
    for seed_i in range(10,101,10):    
        X_wine_train, X_wine_test, y_wine_train, y_wine_test = \
        train_test_split(winefeatures_scaled.iloc[:,0:13], 
                         winefeatures_scaled.iloc[:,13], 
                        test_size=0.3, random_state = seed_i)
        generic_model = model_algo.fit(X_wine_train,y_wine_train)
        predicted_wine = generic_model.predict(X_wine_test)
        accuracy_all[seed_i] = accuracy_score(y_wine_test,predicted_wine)
    cross_validated_accuracy = np.mean(accuracy_all)
    return (cross_validated_accuracy)

wine_cross_validate(KNeighborsClassifier(n_neighbors = 3)) #95.92


############## Logistic Regression ###############################3

diabetes_data = pd.read_csv("E:\\Python Class\\Data\\diabetes_data.csv")
X_train, X_test, y_train, y_test = \
    train_test_split(diabetes_data.iloc[:,0:8], 
                     diabetes_data.iloc[:,8], 
                    test_size=0.3, random_state = 30)
diabetes_logit_model = LogisticRegression().fit(X_train,y_train)
diabetes_logit_prob = diabetes_logit_model.predict_proba(X_test)
diabetes_predict_logit = np.zeros(len(diabetes_logit_prob))

diabetes_predict_logit[diabetes_logit_prob[:,1] > 0.5] = 1
pd.crosstab(y_test,diabetes_predict_logit,
            rownames = ["Actual Diagnosis"], 
            colnames = ["Predicted Diagnosis"])
accuracy_score(y_test,diabetes_predict_logit)
(55+29)/(55+27+7+29)
# decreasing the cutoff to increase sensitivity (increase True Positive Rate)
diabetes_predict_logit = np.zeros(len(diabetes_logit_prob))
diabetes_predict_logit[diabetes_logit_prob[:,1] > 0.3] = 1
pd.crosstab(y_test,diabetes_predict_logit,
            rownames = ["Actual Species"], 
            colnames = ["Predicted Species"])
accuracy_score(y_test,diabetes_predict_logit)

# increase  the cutoff to increase specificity (decrease Fase positive rate)
diabetes_predict_logit = np.zeros(len(diabetes_logit_prob))
diabetes_predict_logit[diabetes_logit_prob[:,1] > 0.7] = 1
pd.crosstab(y_test,diabetes_predict_logit,
            rownames = ["Actual Species"], 
            colnames = ["Predicted Species"])
accuracy_score(y_test,diabetes_predict_logit)

fpr, tpr, thresholds = roc_curve(y_test, diabetes_logit_prob[:,1])
roc_auc = auc(fpr, tpr) # 0.788 AUC which is good
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")