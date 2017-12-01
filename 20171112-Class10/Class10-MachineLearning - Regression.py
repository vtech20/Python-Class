# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 11:44:07 2017

@author: Vignesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import statsmodels.formula.api as smf


catdata = pd.read_csv("E:\\Python Class\\20171112-Class10\\cats.csv")

plt.scatter("Bwt","Hwt",data=catdata)
plt.xlabel("Body weight KG")
plt.ylabel("Heart Weight G")

#step3: Correlation Analysis
#Covariance helps measure the relationship between variables
#However covariance is dependent on the scale of data
#Correlation is scaled covariance which ranges between -1 to 1
### -1 : Strong negative Correlation
### +1 : Strong positive Correlation
###  0 : Weak Correlation ( no relationship)
#proceed with regression model whne you see reasonably stron correlation

np.mean(catdata["Bwt"])
np.var(catdata["Bwt"])
np.std(catdata["Bwt"])
catdata["Bwt"].cov(catdata["Hwt"])
catdata["Bwt"].corr(catdata["Hwt"]) #Strong Positive Correlation

##Step4: Build Regression Model
#using Least Squares method
#import statsmodels.formula.api as smf
#Formula: DV ~ IDV
#ols = least square method
cats_simple_linear_model = smf.ols(formula = 'Hwt ~ Bwt', data = catdata).fit()
cats_simple_linear_model.summary()

Bwt = 3.7
predicted_hwt = 4.0341*Bwt - 0.3567

actual_hwt = 11
#Deviation
abs(actual_hwt - predicted_hwt)/actual_hwt
