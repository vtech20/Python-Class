# -*- coding: utf-8 -*-

# Step 1: Identify Dependent Variable (DV) and Independent Variables(IDVs)
# Step 2: Visualize the relationship between DV and IDVs
  # watch out for curvy relationship
# Step 3: Do a correlation analysis between IDVs and DV
   # Correlation between IDV and DV should be close to 1 or -1 (> 0.5 or < -0.5)
   # Multi Collinearity: IDVs should have weak correlation => Feature Selection
# Step 4: Build regression
   # Do a training-test split
   # R2 range will be between 0 and 1;  close to 1 is agood model
     # Look for Adj R2 in case of multiple IDVs model
    # p values (range: 0 to 1) should be less than 0.05  
# Step 5: Evaluate model fitness
    # Mean Absolute Percentage Error
    # Median Absolute Percentage Error
    # Cross Validation for different training-test split
# Step 6: Go live and start predicting for new data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

############### Cats Data ###########################################
catdata = pd.read_csv("data\\cats.csv")

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
#predicted_hwt = 4.0341*Bwt - 0.3567 # Y = mX + C

# When body weight increases by 1kg, heart weight will increase by 4.0341 grams
# When body weight is zero, heart weight eill be -0.3567 - doesn't make sense
  # intercepts are not always interpretable
Bwt = 3.7
actual_hwt = 11
predicted_hwt = 4.0341*Bwt - 0.3567 # Y = mX + C
#% Deviation
abs(actual_hwt - predicted_hwt)/actual_hwt # 32.4% deivation


# R-squared is the standard metric to measure goodness of fit
# R-squared will be between 0 and 1
# 0 - Bad model
# 1 - ideal model
# R-squared = 0.647 # fiarly ok model fit


# only IDVs with p values less than 0.05 should be included
# p value should be less than 0.05 for the IDV to be significant
# Bwt is significant
# Intercept is insignificant - can be ignored
predicted_hwt = 4.0341*Bwt
# Regression does a hypothesis testing 
# H0 (Null hypotheses): There is no relationship between IDV and DV
# if p value is greater than 0.95, I accept null hypothesis with 95% confidence
# if p value is less than 0.05, I reject null hypothesis with 95% confidence 
# I reject the null hypotheses that there is no relationship

# how accurate is the model?
training_as_test = catdata.iloc[:,0:2] # removing Hwt from data
# predicting heart weight using the model equation
predicted_hwt_trdata = 4.0341*training_as_test["Bwt"] - 0.3567
# use the predict function
predicted_hwt_trdata = cats_simple_linear_model.predict(training_as_test)

percent_error = (catdata["Hwt"] - predicted_hwt_trdata)/catdata["Hwt"] 
abs_percent_error = abs(catdata["Hwt"] - predicted_hwt_trdata)/catdata["Hwt"] 

model_eval = pd.DataFrame({'Bwt':catdata["Bwt"],
                            'Actual Hwt':catdata["Hwt"],
                           'Predicted Hwt':predicted_hwt_trdata,
                           'Percetage Error':percent_error,
                          'Abs Percetage Error':abs_percent_error})
# if the error sign is used, positives and negatives will cancel out showing a less error
# this metric is not used
#np.mean(model_eval['Percetage Error'])

# MEAN ABSOLUTE PERCENTAGE ERROR (MAPE)
np.mean(model_eval['Abs Percetage Error']) # 11.2% error or 89.8% accurate

# Model fitted line
plt.scatter(model_eval['Bwt'],model_eval['Actual Hwt'])
plt.scatter(model_eval['Bwt'],model_eval['Predicted Hwt'],c = "red")
plt.xlim([0,5])
plt.ylim([-1,18]) 
# red line will touch Hwt = -0.3567 (intersept) when Bwt = 0

######### Training - Test Split ########################################
# Build the model on training data and evaluate the model on test data
# 70% for training  - 30% for test

# random sampling 
aa = np.arange(100)
np.random.seed(12345) # fixing the randomness for Reproduciblabilty
# random seed could be any integer
seventy_percent_data = np.random.choice(aa,70,replace=False) # random sampling without replacement
np.in1d(aa,seventy_percent_data) # values in aa which are also present in seventy_percent_data : boolean
~np.in1d(aa,seventy_percent_data) # values in aa which are NOT present in seventy_percent_data : boolean
remaining_thirty_percentage = aa[~np.in1d(aa,seventy_percent_data)]

all_rows = np.arange(144)
all_rows = np.arange(catdata.shape[0])
no_tr_samples = 101
no_tr_samples = round(0.7*catdata.shape[0])
no_te_samples = 43
no_te_samples = round(0.3*catdata.shape[0])
# 101 observations for training and 43 for testing
np.random.seed(10)
training_samples = np.random.choice(all_rows,no_tr_samples,replace=False)
test_samples = all_rows[~np.in1d(all_rows,training_samples)]

cat_training_data = catdata.iloc[training_samples,:] # 70% of cat data for training
cat_test_data = catdata.iloc[test_samples,:] #30% of cat data for test

# Build a linear model using training data
cats_simple_linear_model = smf.ols(formula = 'Hwt ~ Bwt', data = cat_training_data).fit()
cats_simple_linear_model.summary()
# Hwt = 4.1707*Bwt - 0.7076
# R-squared = 0.696
# p value significant for Bwt and insignificant for intercept

# Evaluate the model using test data
# MAPE
cat_test_data2 = cat_test_data.copy()
del cat_test_data2['Hwt']
predicted_hwt_test_data = cats_simple_linear_model.predict(cat_test_data2)

model_eval_test_data = pd.DataFrame({'Bwt':cat_test_data["Bwt"],
                            'Actual Hwt':cat_test_data["Hwt"],
                           'Predicted Hwt':predicted_hwt_test_data})
np.mean(abs(model_eval_test_data['Actual Hwt'] - model_eval_test_data['Predicted Hwt'])/model_eval_test_data['Actual Hwt'])
# 14.34% MAPE on test data


########## Weight Gain Data ###################################################
wgdata = pd.read_csv("data\wg.csv")
plt.scatter("metmin","wg",data = wgdata)
plt.xlabel("Activity")
plt.ylabel("Weight Gain")
wgdata["metmin"].cov(wgdata["wg"])
wgdata["metmin"].corr(wgdata["wg"]) # Significantly strong negative correlation

#DV: wg
#IDV: metmin

wgdata["metmin"].corr(wgdata["wg"]) # strong negative correlation
# as activity increases, weight gain decreases

# wg = m * metmin + intercept
wg_linear_model= smf.ols("wg ~ metmin",data = wgdata).fit()
wg_linear_model.summary()
# wg = 54.16 - 0.0189*metmin

# Clean the data (remove NAs)
wgclean = wgdata.dropna()
# Divide wg data to training and test
all_rows = np.arange(wgclean.shape[0])
no_tr_samples = round(0.7*wgclean.shape[0]) # 179 for training
no_te_samples = round(0.3*wgclean.shape[0]) #77 for testing
np.random.seed(10)
training_samples = np.random.choice(all_rows,no_tr_samples,replace=False)
test_samples = all_rows[~np.in1d(all_rows,training_samples)]
wg_training_data = wgclean.iloc[training_samples,:]
wg_test_data = wgclean.iloc[test_samples,:]
# Build linear model on training data
wg_linear_model = smf.ols('wg ~ metmin', data = wg_training_data).fit()
wg_linear_model.summary()
  # R-squared = 0.823
  # are the p values - coeff and intercept both are significant
  # linear equation
  # wg = -0.0183*metmin + 52.8682
# Test on test data
wg_test_data2 = wg_test_data.copy()
del wg_test_data2["wg"]
predicted_wg = wg_linear_model.predict(wg_test_data2)
wg_abs_percent_error = abs(wg_test_data["wg"] - predicted_wg)/wg_test_data["wg"]
np.mean(wg_abs_percent_error) # 32.8% error
# Median Absolute Percentage Error
np.median(wg_abs_percent_error) # 21.1% error

def MAPE(actual,predicted):
    abs_percent_diff = abs(actual-predicted)/actual
    # 0 actual values might lead to infinite errors
    # replacing infinite with nan
    abs_percent_diff = abs_percent_diff.replace(np.inf,np.nan)
    median_ape = np.nanmedian(abs_percent_diff)
    mean_ape = np.mean(abs_percent_diff)
    mape = pd.Series([mean_ape,median_ape],index = ["Mean_APE","Median_APE"])
    return mape
    
MAPE(wg_test_data["wg"],predicted_wg)
# Mean APE = 32.8%
# Median APE = 21.1%
# Visualizing the model fit
wg_training_as_test = wg_training_data.copy()
del wg_training_as_test["wg"]
wg_training_as_test["predicted_wg_tr"] = wg_linear_model.predict(wg_training_as_test)
plt.scatter("metmin","wg",data = wg_training_data)
plt.scatter("metmin","predicted_wg_tr",data = wg_training_as_test,c="red")

############# Simple Non Linear Regression #####################
# Model fit is not great. Relationship is not linear
# more than 1st order relationship might be needed for a better fit
# y = mX  + C #1st order (linear)
# y = aX^2 + bX + C # 2nd order
# y = aX^3 + bX^2 + cX +  D #3rd order
# Use 2nd order only if thee is a dare necessity
# Never go beyond 2nd order

# wg = a*metmin + b*metmin^2 + intercept

wg_non_linear_model = smf.ols('wg ~ metmin + np.power(metmin,2)', 
                              data = wg_training_data).fit()
wg_non_linear_model.summary()
# wg = -0.0577*metmin + 0.00009715*metmin^2 + 89.7083
# R-squared = 0.969
# p values are significant
predicted_wg_nonlin = wg_non_linear_model.predict(wg_test_data2)
MAPE(wg_test_data["wg"],predicted_wg_nonlin)
# Mean APE = 10%
# Median APE = 7.3%
wg_training_as_test["predicted_wg_tr_nonlin"] = wg_non_linear_model.predict(wg_training_as_test)
plt.scatter("metmin","wg",data = wg_training_data)
plt.scatter("metmin","predicted_wg_tr_nonlin",data = wg_training_as_test,c="red")
plt.scatter("metmin","predicted_wg_tr",data = wg_training_as_test,c="yellow")

# Instead of building a non linear model on raw data
 # data can be transformed to a log scale and model can be linear
########## Data Transformations #################################
wgclean["log_wg"] = np.log(wgclean["wg"])
plt.scatter("metmin","log_wg",data = wgclean)

# Log-Linear model
# Linear-Log
# Log-Log



############## Multiple Regression ######################################3
# More than 1 IDV
cement_data = pd.read_csv("data/cement.csv")

## Step 1:
# DV: y - heat evolved
# IDVs: x1,x2,x3,x4 - composition of key ingedients

## Step 2: Visualization
plt.scatter('x1','y',data=cement_data)
plt.xlabel("x1")
plt.ylabel("y")

plt.scatter('x2','y',data=cement_data)
plt.xlabel("x2")
plt.ylabel("y")

plt.scatter('x3','y',data=cement_data)
plt.xlabel("x3")
plt.ylabel("y")

plt.scatter('x4','y',data=cement_data)
plt.xlabel("x4")
plt.ylabel("y")

axes = pd.tools.plotting.scatter_matrix(cement_data, alpha=0.5)

## Step 3: Correlation
cement_data['x1'].corr(cement_data['y']) # 0.73
cement_data['x2'].corr(cement_data['y']) # 0.81
cement_data['x3'].corr(cement_data['y']) # -0.53
cement_data['x4'].corr(cement_data['y']) # -0.82
corr_matrix = cement_data.corr()

# Step 4: 
cementmodel = smf.ols(formula = 'y ~ x1 + x2 + x3 + x4',
                      data = cement_data).fit()
cementmodel.summary()

# y = 1.5511*x1 + 0.5102*x2 + 0.1019*x3 - 0.1441*x4 + 62.4054
# Adj R-squared: 0.974
# Good model fit

# all p valuea are insignificant
# Multi Collinearity
# Independent variables should not be correlated
 # Each of them should be uniquely explaining DV
# x1 and x3 are highly correlated; one of them is good enough
# x2 and x4 are highly correlated; one of them is good enough
plt.scatter('x1','x3',data = cement_data)
plt.scatter('x2','x4',data = cement_data)

############# FEATURE SELECTION 
 # Selecting only features which are not correlated with each other
# x1, x2
x12_model = smf.ols(formula = 'y ~ x1 + x2', data = cement_data).fit()
x12_model.summary()
# Adj R2: 0.974
# p valuea re significanr

# x1, x3
x13_model = smf.ols(formula = 'y ~ x1 + x3', data = cement_data).fit()
x13_model.summary()
Adj R2: 0.458
# p values are  insignificant
# Since x1 and x3 carry the same information p value is insignificant
# Since information carried in x1 and x3 alone is not sufficient, R2 is poor

# x1, x4
x14_model = smf.ols(formula = 'y ~ x1 + x4', data = cement_data).fit()
x14_model.summary()
# Adj R2: 0.967
# p values are significant

# x2, x3
x23_model = smf.ols(formula = 'y ~ x2 + x3', data = cement_data).fit()
x23_model.summary()
# Adj r2: 0.816
# p values are significant

# x3, x4
x34_model = smf.ols(formula = 'y ~ x3 + x4', data = cement_data).fit()
x34_model.summary()
# Adj R2: 0.922
# p values are significant

# I will finalize on x1, x2
#y = 1.4683*x1 + 0.6623*x2 + 52.5773

################## AVailability #############################################
availability = pd.read_csv("data/Availability.csv")
# Step 1:
# DV: Availability
# IDV: Bid Price, Spot Price

# Step 2:
plt.scatter("Bid","Availability",data = availability)
# As bid price increases availability increases
plt.scatter("Spotprice","Availability",data = availability)

# Step 3:
corr_matrix = availability.corr()
# Availability vs Bid: 0.62 (decent positive correlation)
# Availability vs Spotprice: -0.17 (weak negative correlation)

# Step 4: Model Building
# 70-30 Training Test Split
# 560 obs for training and 240 observations for testing
all_rows = np.arange(800)
np.random.seed(10)
training_samples = np.random.choice(all_rows,int(0.7*800),replace = False)
test_samples = all_rows[~np.in1d(all_rows,training_samples)]
avail_training_data = availability.iloc[training_samples,:]
avail_test_data = availability.iloc[test_samples,:]
avail_test_data2 = avail_test_data.copy()
del avail_test_data2["Availability"]

# Simple Linear Model with Bid Price as IDV
avail_lin_model_bid = smf.ols('Availability ~ Bid', data = avail_training_data).fit()
avail_lin_model_bid.summary()
# R2: 0.420
# p values are signigicant
# Availability = 49.4559*Bid - 0.6167
avail_test_data["predicted_avail_lin_bid"] = avail_lin_model_bid.predict(avail_test_data2)
MAPE(avail_test_data["Availability"],avail_test_data["predicted_avail_lin_bid"])
# 173% Mean APE, 27.4% Median APE

# Simple Linear Model with Spot Price as IDV
avail_lin_model_spot = smf.ols('Availability ~ Spotprice', 
                               data = avail_training_data).fit()
avail_lin_model_spot.summary()
# R2: 0.024
# p values are signigicant
# Availability = -20.3*Spotprice + 1.048
avail_test_data["predicted_avail_lin_spot"] = avail_lin_model_spot.predict(avail_test_data2)
MAPE(avail_test_data["Availability"],avail_test_data["predicted_avail_lin_spot"])
# 291% Mean APE, 38.9% Median APE

# Multiple Linear Model with Bid Price and Spot Price as IDV
avail_multi_linear_model = smf.ols('Availability ~ Spotprice + Bid', 
                                   data = avail_training_data).fit()
avail_multi_linear_model.summary()
# Adj R2: 0.627
# p values are significant
# Availability = 64.89*Bid - 65.48*Spotprice + 0.4940
avail_test_data["predicted_avail_multi_lin"] = avail_multi_linear_model.predict(avail_test_data2)
MAPE(avail_test_data["Availability"],avail_test_data["predicted_avail_multi_lin"])
# 149% Mean APE, 26.8% Median APE

# Multiple Non Linear Model with 2nd order for Bid Price
avail_multi_nonlin_model = smf.ols('Availability ~ Spotprice + Bid + np.power(Bid,2)', 
                                   data = avail_training_data).fit()
avail_multi_nonlin_model.summary()
# Adj R2: 0.8
# p values are significant
# Availability = 373.8*Bid - 6001*Bid^2 - 58*Spotprice - 3.5154
avail_test_data["predicted_avail_multi_nonlin"] = avail_multi_nonlin_model.predict(avail_test_data2)
MAPE(avail_test_data["Availability"],avail_test_data["predicted_avail_multi_nonlin"])
# 120% Mean APE, 18.8% Median APE

############ cross validation #################################
# build and evaluate model for multiple training-test combination
mean_ape_all = []
median_ape_all = []
for seed_i in range(10,101,10): # Seeds 10,20,....,100
    all_rows = np.arange(800)
    np.random.seed(seed_i)
    # Training Test Split
    training_samples = np.random.choice(all_rows,int(0.7*800),replace = False)
    test_samples = all_rows[~np.in1d(all_rows,training_samples)]
    avail_training_data = availability.iloc[training_samples,:]
    avail_test_data = availability.iloc[test_samples,:]
    avail_test_data2 = avail_test_data.copy()
    del avail_test_data2["Availability"]
    # Building Model
    avail_multi_nonlin_model = smf.ols('Availability ~ Spotprice + Bid + np.power(Bid,2)', 
                                   data = avail_training_data).fit()
    # Predicting Availability using Built model
    avail_test_data["predicted_avail_multi_nonlin"] = avail_multi_nonlin_model.predict(avail_test_data2)
    # Evaluating the model
    mape = MAPE(avail_test_data["Availability"],avail_test_data["predicted_avail_multi_nonlin"])
    mean_ape_all.append(mape["Mean_APE"])
    median_ape_all.append(mape["Median_APE"]) 
    
np.mean(mean_ape_all) # 268%
np.mean(median_ape_all) #17.7%


