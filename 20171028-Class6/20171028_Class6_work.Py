# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 09:31:38 2017

@author: Vignesh
"""


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
wg_data = pd.read_csv("E:\\Python Class\\20171028-Class6\\wg.csv")
wg_data.columns
wg_data['wg'].isnull().sum()
wg_data['metmin'].isnull().sum()

s1=wg_data['wg'].isnull()
s2=wg_data['metmin'].isnull()
condx=s1 ==False
condy=s2==False
s3= wg_data.loc[condx | condy,:]
s3.shape[0]

wg_data.loc[(wg_data["wg"] > np.mean(wg_data["wg"]))].shape[0]

avg_data = wg_data.loc[(wg_data["wg"] > np.mean(wg_data["wg"])),:]
avg_data.groupby(['Shift']).size()

wgClean = wg_data.dropna()
print(wgClean)

############Vizualization#######
###########3Histogram##################
# Vizualize the Distribution of Data
hist_plt = wgClean["wg"].plot.hist()
hist_plt.set_xlabel("Weight Gain")
hist_plt.set_ylabel("No of Observations")
hist_plt.set_title("Distribution of Weight Gain")
hist_plt.set_xlim([5,45]) #To limitize the X range

#####using matplotlib########
plt.hist(wgClean["wg"],color="black")  
plt.hist(wgClean["wg"],bins=[0,10,20,30,40,50,60,70])
plt.hist(wgClean["wg"],bins=[0,20,40,60,80])  

plt.hist(wgClean["wg"],color="black")
plt.xlabel("Weight Gain")
plt.ylabel("No of Observations")
plt.title("Distribution of Weight Gain")
plt.xlim([5,45]) #To limitize the X range

####Custom Binning###########
wgClean['wg'].plot.hist(bins=[0,20,40,60,80])

############Box Plot ############

a1 = np.random.normal(10,1,100)
plt.hist(a1)
np.mean(a1)
np.median(a1)

a2 = np.append(a1,100000)
plt.hist(a2)
np.mean(a2)
np.median(a2)

wgClean['wg'].plot.box()
wgClean['wg'].describe()
q3 = np.percentile(wgClean["wg"],75)
q1 = np.percentile(wgClean["wg"],25)
iqr = q3 - q1
uwl = q3 + 1.5*iqr
lwl = q1 - 1.5*iqr
np.min(wgClean["wg"])
np.max(wgClean["wg"])


np.percentile(wgClean["wg"],100)

############Comparing the Disributions###
wgClean.boxplot(column="wg",by="Gender")
wgClean.boxplot(column="wg",by="Shift")

###########Scatter Plot###############
sc_plt = wgClean.plot.scatter("metmin","wg")
sc_plt.set_xlabel("Activities")
sc_plt.set_ylabel("Weight Gain")

#######Matplot lib################

plt.scatter(wgClean['metmin'],wgClean['wg'])
#plotting Columns of a DataFrame####
plt.scatter("metmin","wg",data=wgClean)

#Different Color for Male and Female Observations
wgClean["color"]= "red"
wgClean.loc[wgClean["Gender"]=="M","color"] = "blue"
plt.scatter("metmin","wg",data=wgClean,c="color")

                
