# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 23:12:53 2017

@author: Vignesh
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

availablityData = pd.read_csv('E:\\Python Class\\Data\\Availability.csv')
print(availablityData)



# Step 1 : Identify dv and idv

# Availablity : Dv

# Bid Price : Idv

# Scatter plot : Bid Proce vs Availablity

plt.scatter('Bid','Availability',data=availablityData)
plt.xlabel("Bid Value") # IDV to x 
plt.ylabel("Availablity") # DV to y 

corr_matrix = availablityData.corr()

availablityData["Bid"].corr(availablityData["Availability"]) # 0.62035 - Decent positive correlation

all_rows = np.arange(availablityData.shape[0])
# Training - 70 % and test 30 % of the 800 records
trainingRecordCount = round(0.7 * availablityData.shape[0]) # 560 for training
testingRecordCount = round(0.3 * availablityData.shape[0]) # 240 for test

np.random.seed(100)
training_samples = np.random.choice(all_rows,trainingRecordCount,replace= False) # row index of data to be used for training
testing_samples = all_rows[~ np.in1d(all_rows ,training_samples)] 
                           
availablityData_training = availablityData.iloc[training_samples,:]
availablityData_testing = availablityData.iloc[testing_samples,:]  

availablityData_training_model = smf.ols(formula = 'Availability ~ Bid',data=availablityData_training).fit()
availablityData_training_model.summary()    

# R-squared is the standard metric to measure goodness of fit

# R-squared will be between 0 and 1

# R-squared = 0.408 # bad model fit

# P value is 0 . IDv - Bid is significant

# availablity = 51.6434 * Bid - 0.6559 



def MAPE(actual,predicted):
    abs_percent_diff = abs(actual-predicted)/actual
    # 0 actual values might lead to infinite errors
    # replacing infinite with nan
    abs_percent_diff = abs_percent_diff.replace(np.inf,np.nan)
    median_ape = np.nanmedian(abs_percent_diff)
    mean_ape = np.mean(abs_percent_diff)
    mape = pd.Series([mean_ape,median_ape],index = ["Mean_APE","Median_APE"])
    return mape
    
# 5.1: Copy test dataset to a new dataset with index, dv and idv columns

availablityData_testingcopy2 = availablityData_testing.copy()
del availablityData_testingcopy2["Availability"]

#5.2: Create dataframe by applying predict method of model on the dataset created in the previous step . 

#    This is predicted testing data for dependent variable

predicted_Availability = availablityData_training_model.predict(availablityData_testingcopy2) 

MAPE(availablityData_testing["Availability"],predicted_Availability)   

#Mean Ape is 2.481
#Median Ape 0.257  #25.7% abs error
abs_percent_diff = abs(availablityData_testing["Availability"]-predicted_Availability)/availablityData_testing["Availability"]
abs_percent_diff = abs_percent_diff.replace(np.inf,np.nan)
model_test_data = pd.DataFrame({'Bid':availablityData_testing["Bid"],
                            'Availability':availablityData_testing["Availability"],
                           'Predicted Avail':predicted_Availability,
                           'Abs Percetage Error':abs_percent_diff})

plt.scatter('Bid','Availability',data = model_test_data)
plt.scatter("Bid","Predicted Avail",data = model_test_data,c="red")

#########################################################################################
#Spot and Availability


plt.scatter('Spotprice','Availability',data=availablityData)
plt.xlabel("Spot Price") # IDV to x 
plt.ylabel("Availablity")

#Y value is increasing even for the same X value

availablityData["Spotprice"].corr(availablityData["Availability"])   # - 0.172 Low correlaton

avail_spot_model = smf.ols(formula = 'Availability ~ Spotprice',data=availablityData_training).fit()
avail_spot_model.summary()

#R-Squared is 0.030 ### Bad Fit
#P is significant for both

predicted_Availability_1 = avail_spot_model.predict(availablityData_testingcopy2) 
MAPE(availablityData_testing["Availability"],predicted_Availability_1)  

#Mean APE is 4.097      #409% Mean abs error              
#Median is 0.395  #39.5% Median Abs error
abs_percent_diff = abs(availablityData_testing["Availability"]-predicted_Availability_1)/availablityData_testing["Availability"]
abs_percent_diff = abs_percent_diff.replace(np.inf,np.nan)
model_test_data_1 = pd.DataFrame({'Spot price':availablityData_testing["Spotprice"],
                            'Availability':availablityData_testing["Availability"],
                           'Predicted Avail':predicted_Availability_1,
                           'Abs Percetage Error':abs_percent_diff})

plt.scatter('Spot price','Availability',data = model_test_data_1)
plt.scatter("Spot price","Predicted Avail",data = model_test_data_1,c="red")

############Bad Fit Model###########
############ Multiple Non Linear####################################

avail_non_linear_model = smf.ols('Availability ~ Spotprice +Bid + np.power(Bid,2)', data = availablityData_training).fit()
avail_non_linear_model.summary()
#R Squared is 0.789 which is fairly good
#P is significant for all
predicted_avail_nonlin = avail_non_linear_model.predict(availablityData_testingcopy2)

MAPE(availablityData_testing["Availability"],predicted_avail_nonlin) 
#Mean_APE      1.771
#Median_APE 0.176  # 17.6% median ape

abs_percent_diff_1 = abs(availablityData_testing["Availability"]-predicted_avail_nonlin)/availablityData_testing["Availability"]
abs_percent_diff_1 = abs_percent_diff_1.replace(np.inf,np.nan)

model_test_data_2 = pd.DataFrame({'Bid':availablityData_testing["Bid"],
                           'Spot price':availablityData_testing["Spotprice"],
                            'Availability':availablityData_testing["Availability"],
                           'Predicted Avail':predicted_avail_nonlin,
                           'Abs Percetage Error':abs_percent_diff_1})

plt.scatter('Bid','Availability',data = model_test_data_2)
plt.scatter("Bid","Predicted Avail",data = model_test_data_2,c="red")

#################Multiple Linear Regression#########################################

## Step 1:
# DV: Availability -
# IDVs: Bid Price & Spot Price

## Step 2: Visualization
plt.scatter('Bid','Availability',data=availablityData)
plt.xlabel("Bid")
plt.ylabel("Availability")

plt.scatter('Spotprice','Availability',data=availablityData)
plt.xlabel("Spot price")
plt.ylabel("Availability")

availablityData['Bid'].corr(availablityData['Availability']) # 0.62
availablityData['Spotprice'].corr(availablityData['Availability']) # -0.17

#some what ok correlation

# Step 4: 
availablity_multi_model = smf.ols(formula = 'Availability ~ Bid + Spotprice',
                      data = availablityData_training).fit()
availablity_multi_model.summary()


#Adj. R-squared:                  0.621
#All P values are significant
#Availability = 67.62*bid -66.10*spotprice +0.455

predicted_avail_multi = availablity_multi_model.predict(availablityData_testingcopy2)

MAPE(availablityData_testing["Availability"],predicted_avail_multi) 

#Mean_APE      2.008276
#Median_APE    0.274 #27.4% abs error

abs_percent_diff_2 = abs(availablityData_testing["Availability"]-predicted_avail_multi)/availablityData_testing["Availability"]
abs_percent_diff_2 = abs_percent_diff_1.replace(np.inf,np.nan)

model_test_data_3 = pd.DataFrame({'Bid':availablityData_testing["Bid"],
                           'Spotprice':availablityData_testing["Spotprice"],       
                            'Availability':availablityData_testing["Availability"],
                           'Predicted Avail':predicted_avail_multi,
                           'Abs Percetage Error':abs_percent_diff_2})


plt.scatter('Bid','Availability',data = model_test_data_3)
plt.scatter("Spotprice","Availability",data = model_test_data_3,c="red")
plt.scatter("Bid","Predicted Avail",data = model_test_data_3,c="cyan")


#################Cross Validation#############################

for seed_i in range(10,101,10):
    all_rows = np.arrange(800)
    np.random.seed(seed_i)
    training_samples = np.random.choice(all_rows,int(0.7*800), replace = False)
    test_samples = all_rows[~np.in1d(all_rows,training_samples)]
    avail_train_data = availablityData.iloc[training_samples,:]
    avail_test_data = availabilityData.iloc[test_samples,:]
    avail_test_data2 = avail_test_data.copy()
    del avail_test_data2["Availability"]
    avail_multi_nonlin_model = smf.ols('Availability ~ Spotprice + Bid + np.power(Bid,2)',data = avail_train_data).fit()
    