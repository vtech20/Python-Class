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

airqualityclean.describe()
#Get the histogram distribution of Solar.R 
hist_plt = airquality["Solar.R"].plot.hist()
hist_plt.set_xlabel("Solar.R")
hist_plt.set_ylabel("Number of observations")

#2Get the boxplot distribution of temperature
plt.boxplot(airquality["Temp"])
Q3 = np.percentile(airquality["Temp"],75) 
Q1 = np.percentile(airquality["Temp"],25) 
IQR = Q3 - Q1
print(IQR)
UWL = Q3 + 1.5*IQR
LWL = Q1 - 1.5*IQR
print(UWL)
print(LWL)

#3Generate a scatter plot between temperature and solar.R 
sct_plt = airquality.plot.scatter("Temp","Solar.R")
sct_plt.set_xlabel("Temparature")
sct_plt.set_ylabel("Solar Radiation")

plt.scatter("Temp",'Solar.R',data = airquality,c = "red")
plt.scatter("Temp",'Solar.R',data = airquality,c = "Blue")

#4
airquality.index = "1973"+ "-"+ airquality["Month"].astype(str) + "-"+ airquality["Day"].astype(str)
airquality["Solar.R"].plot.line()
airquality["Temp"].plot.line()

#or
date_string = "1973"+ "-"+ airquality["Month"].astype(str) + "-"+ airquality["Day"].astype(str)
date_datetime = pd.to_datetime(date_string)
airquality.index = date_datetime

airquality["Solar.R"].plot.line()
airquality["Temp"].plot.line()

#or
date_string = "1973-"+ airquality["Month"].map(str) + "-"+ airquality["Day"].map(str)
date_datetime = pd.to_datetime(date_string)
airquality.index = date_datetime

airquality["Solar.R"].plot.line()
airquality["Temp"].plot.line()

date_string_ind = pd.Series([
                             "06,01,1989",
                             "24,08,1975",
                             "19,06,1980"])
pd.to_datetime(date_string_ind,format = "%d,%m,%Y")
#############MT Cars
mtcars_data = pd.read_csv("E:\\Python Class\\Assignments\\mtcars.csv")
#1
mtcars_data.boxplot(column='mpg',by='am')
#2
mtcars_data.boxplot(column='mpg',by=['am','gear'])
#3
mtcars_data.plot.scatter('mpg','wt')


#conda install seaborn




 