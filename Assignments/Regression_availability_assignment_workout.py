# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 08:46:00 2017

@author: Vignesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

availdata = pd.read_csv("E:\\Python Class\\Data\\Availability.csv")

# Step 1:
#DV: Availability
#IDV: Bid
#2. Visualize the relationship between Bid Price and Availability using scatter plot 
#Step2
plt.scatter("Bid","Availability",data=availdata)
plt.xlabel("Bid Price")
plt.ylabel("Availability")
#When Bid price increases availability is also increases
#good relationship
#Step3 Check the correlation between Bid proce and availability
#3 Check the correlation between Bid Price and Availability
np.mean(availdata["Bid"]) #0.0241
np.var(availdata["Bid"])  # 1.9349
np.std(availdata["Bid"])  #0.004
availdata["Bid"].cov(availdata["Availability"])  #0.00092
availdata["Bid"].corr(availdata["Availability"])  #0.6203
#Not a good Covariance but a considerable one
#4. Do a Training-Test Split
# Least Squares method
#import statsmodels.formula.api as smf
# Formula: DV ~ IDV
total_rows1 = np.arange(availdata.shape[0])
training_count1 = np.arange(0.7*availdata.shape[0])
test_count1 = np.arange(0.3*availdata.shape[0])
#560 Training Data and #240 Test data
np.random.seed(100)
train_samples1 = np.random.choice(total_rows1,training_count1,replace=False)
test_samples1 = total_rows1[~np.in1d(total_rows1,train_samples1)]