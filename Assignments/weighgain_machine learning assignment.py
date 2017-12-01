# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 08:13:57 2017

@author: Vignesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import statsmodels.formula.api as smf

wgdata = pd.read_csv("E:\\Python Class\\20171028-Class6\\wg.csv")

plt.scatter("metmin","wg",data=wgdata)
plt.xlabel("Activities")
plt.ylabel("Weight Gain")

#step3: Correlation Analysis
#Covariance helps measure the relationship between variables
#However covariance is dependent on the scale of data
#Correlation is scaled covariance which ranges between -1 to 1
### -1 : Strong negative Correlation
### +1 : Strong positive Correlation
###  0 : Weak Correlation ( no relationship)
#proceed with regression model whne you see reasonably stron correlation

np.mean(wgdata["metmin"])
np.var(wgdata["metmin"])
np.std(wgdata["metmin"])
wgdata["metmin"].cov(wgdata["wg"])
wgdata["metmin"].corr(wgdata["wg"]) #Strong negative Correlation

##Step4: Build Regression Model
#using Least Squares method
#import statsmodels.formula.api as smf
#Formula: DV ~ IDV
#ols = least square method
wg_simple_linear_model = smf.ols(formula = 'wg ~ metmin', data = wgdata).fit()
wg_simple_linear_model.summary()

#predicted_wg = 54.1624*metmin - 56.409
#compare
metmin = 1628
predicted_wg = -0.0189*metmin + 54.1624
print(predicted_wg)

#R-Squared is 0.821

