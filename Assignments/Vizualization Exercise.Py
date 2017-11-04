# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 07:37:13 2017

@author: Vignesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

airquality = pd.read_csv("E:\\Python Class\\Assignments\\airquality.csv")
airqualityclean = airquality.dropna()
#Get the histogram distribution of Solar.R 
hist_plt = airqualityclean["Solar.R"].plot.hist()
hist_plt.set_xlabel("Solar.R")
hist_plt.set_ylabel("Number of observations")

#2Get the boxplot distribution of temperature
plt.boxplot(airqualityclean["Temp"])
Q3 = np.percentile(airqualityclean["Temp"],75) 
Q1 = np.percentile(airqualityclean["Temp"],25) 
IQR = Q3 - Q1
print(IQR)
UWL = Q3 + 1.5*IQR
LWL = Q3 - 1.5*IQR
print(UWL)
print(LWL)

#3Generate a scatter plot between temperature and solar.R 
sct_plt = airqualityclean.plot.scatter("Temp","Solar.R")
sct_plt.set_xlabel("Temparature")
sct_plt.set_ylabel("Solar Radiation")

plt.scatter("Temp",'Solar.R',data = airqualityclean,c = "red")
plt.scatter("Temp",'Solar.R',data = airqualityclean,c = "Blue")

#4
airqualityclean['Date'] = "1973"+ airqualityclean["Month"].astype(str) + airqualityclean["Day"].astype(str)


#############MT Cars
mtcars_data = pd.read_csv("E:\\Python Class\\Assignments\\mtcars.csv")
#1
mtcars_data.boxplot(column='mpg',by='vs')
#2
mtcars_data.boxplot(column='mpg',by=['vs','gear'])
#3
mtcars_data.plot.scatter('mpg','wt')

 