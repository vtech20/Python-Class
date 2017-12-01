# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 07:40:41 2017

@author: Vignesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
wgdata = pd.read_csv("E:\\Python Class\\Data\\wg.csv")
wgdataclean = wgdata.dropna()




total_rows = np.arange(wgdataclean.shape[0])
training_count = round(0.7*wgdataclean.shape[0])
test_count = round(0.3*wgdataclean.shape[0])
np.random.seed(100)
tr_samples = np.random.choice(total_rows,training_count,replace=False)
te_samples = total_rows[~np.in1d(total_rows,tr_samples)]

wg_training_data = wgdataclean.iloc[tr_samples,:]
wg_test_data = wgdataclean.iloc[te_samples,:]


plt.scatter("metmin","wg",data = wg_training_data)
plt.xlabel("Activity")
plt.ylabel("Weight Gain")
wg_training_data["metmin"].cov(wg_training_data["wg"])   #-6055.42922
wg_training_data["metmin"].corr(wg_training_data["wg"])  #-0.912858

wg_practise_model = smf.ols(formula='wg ~ metmin',data=wg_training_data).fit()
wg_practise_model.summary()
#R-Squared = 0.833
#p is 0.00 for both Intercept and Metmin
#p is significant for both.

test1 = wg_test_data.loc[:,['Gender','metmin']]
predict_wg_te_wg = wg_practise_model.predict(test1)

error_percent = (wg_test_data["wg"] - predict_wg_te_wg )/wg_test_data["wg"]
abs_error = abs(wg_test_data["wg"] - predict_wg_te_wg )/wg_test_data["wg"]
                
model_eval_test = pd.DataFrame({'metmin' : wg_test_data["metmin"],
                           'Actual wg': wg_test_data["wg"],
                            'Predicted wg':predict_wg_te_wg,
                            'Percentage Error':error_percent,
                            'absolute Percent Error':abs_error})     

np.mean(model_eval_test["absolute Percent Error"])
#incase of extreme variations,people used to take median absolute percentage error
np.median(model_eval_test["absolute Percent Error"])
#41.72% error

#Model fitted Line
plt.scatter(model_eval_test['metmin'],model_eval_test['Actual wg']) 
plt.scatter(model_eval_test['metmin'],model_eval_test['Predicted wg'],c='red') 
plt.xlim([0,5])
plt.ylim([-1,18])   


def MAPE(actual,predicted):
    abs_percent_diff = abs(actual-predicted)/actual
    abs_percent_diff = abs_percent_diff.replace(np.inf,np.nan)
    median_ape = np.median(abs_percent_diff)
    mean_ape = np.mean(abs_percent_diff)
    mape = pd.Series([mean_ape,median_ape],index=['Mean_ape','Median_ape'])
    return mape
    
MAPE(wg_test_data["wg"],predict_wg_te_wg) 

#Model fit is not great, Relationship is not linear
#More than 1st order relationship as the relationship is a curve
# y = mx+c
# y = ax^2 + Bx + c = 2nd order
# Go to the 2nd order onl y if there is a necessity 

wg_non_linear_model = smf.ols('wg ~ metmin + np.power(metmin,2)', data = wg_training_data).fit()
wg_non_linear_model.summary()
#R2 = 0.977
#p values are significant
predicted_wg_1 = wg_non_linear_model.predict(wg_test_data)
MAPE(wg_test_data["wg"],predicted_wg_1) 
abs_error = abs(wg_test_data["wg"] - predicted_wg_1 )/wg_test_data["wg"]
model_eval_nonlin_test = pd.DataFrame({'metmin' : wg_test_data["metmin"],
                           'Actual wg': wg_test_data["wg"],
                            'Predicted wg':predicted_wg_1,
                           'absolute Percent Error':abs_error})   

plt.scatter(model_eval_nonlin_test['metmin'],model_eval_nonlin_test['Actual wg']) 
plt.scatter(model_eval_nonlin_test['metmin'],model_eval_nonlin_test['Predicted wg'],c='red') 
plt.xlim([0,5000])
plt.ylim([-1,18])   
    

####################Data Transformations ##################################
wgdataclean["log_wg"] = np.log(wgdataclean["wg"])
plt.scatter("metmin","log_wg",data = wgdataclean)
plt.xlabel("Activity")
plt.ylabel("log Weight Gain")        

#Try doing the model using log_wg



##################Multiple Regression####################
######More than 1 IDV
cementdata = pd.read_csv("E:\\Python Class\\Data\\cement.csv")

#step1
#DV - Y
#IDV - X1, X2, X3, X4

#step2 - Visualization

plt.scatter('x1','y',data=cementdata)
plt.xlabel('x1')
plt.ylabel('y')

plt.scatter('x2','y',data=cementdata)
plt.xlabel('x2')
plt.ylabel('y')

plt.scatter('x3','y',data=cementdata)
plt.xlabel('x3')
plt.ylabel('y')

plt.scatter('x4','y',data=cementdata)
plt.xlabel('x4')
plt.ylabel('y')

axes = pd.tools.plotting.scatter_matrix(cementdata, alpha=0.5)

#Step 3
cementdata['x1'].corr(cementdata['y']) #0.7307
cementdata['x2'].corr(cementdata['y']) #0.81
cementdata['x3'].corr(cementdata['y']) #-0.53
cementdata['x4'].corr(cementdata['y']) #-0.821
corr_matrix = cementdata.corr()

cement_model = smf.ols(formula = 'y ~ x1 + x2 + x3 + x4',data=cementdata).fit()
cement_model.summary()

#y = 1.5511*x1 + 0.5102*x2 + 0.1019*x3 - 0.1441*x4 + 62.4054
#Good Model fit

# all p values are not significant

#x1
cement_model_x1 = smf.ols(formula = 'y ~ x1',data=cementdata).fit()
cement_model_x1.summary()
  #R2 = 0.534
  #P = 0.05
#x2
cement_model_x2 = smf.ols(formula = 'y ~ x2',data=cementdata).fit()
cement_model_x2.summary()
  #R2 = 0.666
  #P = 0.001
#x3
cement_model_x3 = smf.ols(formula = 'y ~ x3',data=cementdata).fit()
cement_model_x3.summary()
  #R2 = 0.286
  #P = 0.060
#x4
cement_model_x4 = smf.ols(formula = 'y ~ x4',data=cementdata).fit()
cement_model_x4.summary()
  #R2 = 0.675
  #P = 0.001
#x1, x2
cement_model_x1_x2 = smf.ols(formula = 'y ~ x1 + x2',data=cementdata).fit()
cement_model_x1_x2.summary()
  #R2 = 0.974
  #P (X1) = 0.000
  #P (X2) = 0.000
#x1, x3
cement_model_x1_x3 = smf.ols(formula = 'y ~ x1 + x3',data=cementdata).fit()
cement_model_x1_x3.summary()
  #R2 = 0.458
  #P (X1) = 0.037
  #P (X3) = 0.587
#x1, x4
cement_model_x1_x4 = smf.ols(formula = 'y ~ x1 + x4',data=cementdata).fit()
cement_model_x1_x4.summary()
  #R2 = 0.967
  #P (X1) = 0.000
  #P (X3) = 0.000
#x2, x3
#x2, x4


