# -*- coding: utf-8 -*-

# Step 1: Identify Dependent Variable (DV) and Independent Variables(IDVs)
# Step 2: Visualize the relationship between DV and IDVs
# Step 3: Do a correlation analysis between IDVs and DV
# Step 4: Build regression
# Step 5: Evaluate model fitness
# Step 6: Go live and start predicting for new data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

############### Cats Data ###########################################
catdata = pd.read_csv("E:\\Python Class\\Data\\cats.csv")

# Step 1:
#DV: Hwt
#IDV: Bwt
# Hwt = m * Bwt + C
# Sales = a * AdSpend + b * Price + c * StorePromotions + BaseSales


# Step 2: Visualize (Scatter Plot)
plt.scatter("Bwt","Hwt",data=catdata)
plt.xlabel("Body Weight(Kg)")
plt.ylabel("Heart Weight(g)")

# Step 3: Correlation analysis
# Covariance helps measure the relationship between variables
# However covariance is dependent on the scale of data
# Correlation is scaled covariance which ranges between -1 to +1
  # -1: Strong negative correlation
  # +1: Strong positive correlation
  # 0: Weak correlation (no relationship)
# Proceed with regression model noy, when you see reasonably strong correlation
np.mean(catdata["Bwt"])
np.var(catdata["Bwt"])
np.std(catdata["Bwt"])
catdata["Bwt"].cov(catdata["Hwt"])
catdata["Bwt"].corr(catdata["Hwt"]) # Strong positive correlation


## Step 4: Build Regression Model
# Least Squares method
#import statsmodels.formula.api as smf
# Formula: DV ~ IDV
cats_simple_linear_model = smf.ols(formula = 'Hwt ~ Bwt', data = catdata).fit()
cats_simple_linear_model.summary()
Bwt = 3.7
predicted_hwt = 4.0341*Bwt - 0.3567 # Y = mX + C
actual_hwt = 11
#% Deviation
abs(actual_hwt - predicted_hwt)/actual_hwt # 32.4% deivation

# P value should be less than 0.05 for IDV to be significant
#bwt is Significant
#0 - Bad model
#1 - ideal model
#R-Squared = 0.647 #fairly ok model fit

#only IDVs with p values less than 0.05 should be included
#Regression does a null hypothesis
#Null Hypothesis : There is no relationship between IDV and DV
#if p > 0.95, I accept null hypothesis with 95% confidence
#if p < 0.05, I reject null hypothesis with 95% confidence

#How accurate the model?
training_as_test = catdata.iloc[:,0:2]
type(training_as_test)
predict_hwt_trdata = cats_simple_linear_model.predict(training_as_test)

percent_error = (catdata["Hwt"] - predict_hwt_trdata )/catdata["Hwt"]
abs_percent_error = abs(catdata["Hwt"] - predict_hwt_trdata )/catdata["Hwt"]
                        
model_eval = pd.DataFrame({'Bwt' : catdata["Bwt"],
                           'Actual Hwt': catdata["Hwt"],
                            'Predicted Hwt':predict_hwt_trdata,
                            'Percentage Error':percent_error,
                            'absolute Percent Error':abs_percent_error}) 

aa = np.mean(model_eval["absolute Percent Error"]) # 11.2% error or 89.8% accurate

#Model fitted Line
plt.scatter(model_eval['Bwt'],model_eval['Actual Hwt']) 
plt.scatter(model_eval['Bwt'],model_eval['Predicted Hwt'],c='red') 
plt.xlim([0,5])
plt.ylim([-1,18])

########## Training - Test Split ######################
# Build the model on training data and evaluate the model on test data
# 70% for training - 30% for test
round(0.7*144)
round(0.3*144)
#101 observations for training and 43 observations for testing
#random sampling
aa=np.arange(100)
np.random.seed(12345)
seventy_per_data = np.random.choice(aa,70,replace=False) #Random Sampling without replacement
remaining_thirty_percentage = aa[~np.in1d(aa,seventy_per_data)] # Values in aa which is not present in seventy_per_data

all_rows = np.arange(catdata.shape[0])
no_tr_samples = round(0.7*catdata.shape[0])
no_te_samples = round(0.3*catdata.shape[0])
np.random.seed(10)
training_samples = np.random.choice(all_rows,no_tr_samples,replace=False)
test_samples = all_rows[~np.in1d(all_rows,training_samples)]

cat_training_data = catdata.iloc[training_samples,:]
cat_test_data = catdata.iloc[test_samples,:]
#Build Linear Model using Training Data
#Step1
plt.scatter("Bwt","Hwt",data=cat_training_data)
plt.xlabel("Body Weight(Kg)")
plt.ylabel("Heart Weight(g)")
#Step2
np.mean(cat_training_data["Bwt"])
np.var(cat_training_data["Bwt"])
np.std(cat_training_data["Bwt"])
cat_training_data["Bwt"].cov(cat_training_data["Hwt"])
cat_training_data["Bwt"].corr(cat_training_data["Hwt"])
#Step3
cats_simple_linear_model1 = smf.ols(formula = 'Hwt ~ Bwt', data = cat_training_data).fit()
cats_simple_linear_model1.summary()
#Evaluate using Test Data

test1 = cat_test_data.iloc[:,0:2]
type(test1)
predict_hwt_tedata = cats_simple_linear_model1.predict(test1)

percent_error1 = (cat_test_data["Hwt"] - predict_hwt_tedata )/cat_test_data["Hwt"]
abs_percent_error1 = abs(cat_test_data["Hwt"] - predict_hwt_tedata )/cat_test_data["Hwt"]
                        
model_eval_test = pd.DataFrame({'Bwt' : cat_test_data["Bwt"],
                           'Actual Hwt': cat_test_data["Hwt"],
                            'Predicted Hwt':predict_hwt_tedata,
                            'Percentage Error':percent_error1,
                            'absolute Percent Error':abs_percent_error1}) 

np.mean(model_eval_test["absolute Percent Error"])


#Model fitted Line
plt.scatter(model_eval_test['Bwt'],model_eval_test['Actual Hwt']) 
plt.scatter(model_eval_test['Bwt'],model_eval_test['Predicted Hwt'],c='red') 
plt.xlim([0,5])
plt.ylim([-1,18])

########## Weight Gain Data ###################################################
wgdata = pd.read_csv("data\wg.csv")
plt.scatter("metmin","wg",data = wgdata)
plt.xlabel("Activity")
plt.ylabel("Weight Gain")
wgdata["metmin"].cov(wgdata["wg"])
wgdata["metmin"].corr(wgdata["wg"]) # Significantly strong negative correlation

#DV: wg
#IDV: metmin

# wg = m * metmin + intercept

#Clean the data (remove NAs)
#Divide wg data to train and test
#Build Linear model on training data
  # R - Squared
  # are the p values significant
  #linear equation
# test on Test data
  # MAPE
  