# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:53:39 2017

@author: Vignesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

emp_data = pd.read_csv("E:\\Python Class\\Assignments\\ssamatab1\\ssamatab1.csv")
print(emp_data)
type(emp_data.ix[0,"Civilian Labor Force"])

#1. Read the csv file and check the data types. Note that certain columns has numbers with
#commas in between which might have been read as a non-numeric data type. You can't
#just convert the data type; it will then have junk numbers. You have to remove commas.
emp_data_clean = pd.read_csv("E:\\Python Class\\Assignments\\ssamatab1\\ssamatab1.csv",thousands=',')
type(emp_data_clean.ix[0,"Civilian Labor Force"])

#2Which Area had the highest unemployment rate in December 2015?
s_dat = emp_data_clean.loc[(emp_data_clean["Year"] ==2015) & (emp_data_clean["Month"] ==12),:]
s_dat.loc[s_dat["Unemployment Rate"] == np.max(s_dat["Unemployment Rate"]),"Area"]
          
#3Which area had the highest ever unemployment rate and when did that happen?
emp_data_clean.loc[emp_data_clean["Unemployment Rate"] == np.max(emp_data_clean["Unemployment Rate"]),["Year","Month","Area"]]         

#4 Which state had the highest ever unemployment rate and when did that happen?
emp_data_clean.loc[emp_data_clean["Unemployment Rate"] == np.max(emp_data_clean["Unemployment Rate"]),["Year","Month","State FIPS Code"]]                            

#5. Obtain Yearly Unemployment rate by aggregating the data. One way would be to take
#average of unemployment rate column directly. But that's not mathematically right. You
#need to sum up the Unemployed and Civilian labor force by Year and then calculate the
#ratio for calculation of Unemployment rate  

yearly_grp = emp_data_clean.groupby(['Year']) 
civun_data = yearly_grp[["Civilian Labor Force","Unemployment"]].agg(np.sum)
yearly_up_rate = civun_data["Civilian Labor Force"] / civun_data["Unemployment"]

#6 Repeat a similar aggregation as previous point for State Level unemployment rate                

state_grp = emp_data_clean.groupby(['State FIPS Code'])
print(state_grp) 
civun_data1 = state_grp[["Civilian Labor Force","Unemployment"]].agg(np.sum)
state_up_rate = civun_data1["Civilian Labor Force"] / civun_data1["Unemployment"]

#7 Plot the histogram and boxplot of unemployment rate

plt.hist(emp_data_clean["Unemployment Rate"],bins=[0,3,6,9,12,15,18,21,24,27,30])
plt.xlabel("Unemployment Rate")
plt.ylabel("Number of observations")
plt.title("Distribution of Unemployment Rate")

plt.boxplot(emp_data_clean["Unemployment Rate"])

#8 Compare the boxplot distribution of unemployment rate between top 4 states with highest
#civilian labor force
civun_data2 = state_grp[["Civilian Labor Force","State FIPS Code"]].agg(np.max)
civun_data2_sorted = civun_data2.sort_values("Civilian Labor Force",ascending = False).head(4)
civun_data3 = emp_data_clean.loc[(emp_data_clean["State FIPS Code"].isin(civun_data2_sorted["State FIPS Code"])),:]
civun_data3.boxplot(column="Unemployment Rate",by="State FIPS Code")

s = emp_data_clean.loc[(emp_data_clean["State FIPS Code"].isin(1),:]

#9Visualize the relationship between civilian labor force and unemployment rate using
#scatter plot
plt.subplot(141)
#plt.axis([0.0,0.2,0,35])
plt.scatter(emp_data_clean["Civilian Labor Force"],emp_data_clean["Unemployment Rate"])

ur_grp = emp_data_clean.groupby(['Unemployment Rate'])
cv_data = ur_grp[["Civilian Labor Force"]].agg(np.sum)
aa = pd.DataFrame(cv_data)
cv_data['URate'] = aa.index
plt.scatter(cv_data['Civilian Labor Force'],cv_data['URate'])


#sct_plt = wgclean.plot.scatter("metmin","wg")

sct_plot = emp_data_clean.plot.scatter("Civilian Labor Force","Unemployment Rate")
sct_plot.set_xlim(0.0,0.2)

plt.scatter(emp_data_clean["Civilian Labor Force"],emp_data_clean["Unemployment Rate"],emp_data_clean["Year"])

ax1 = plt.subplot(131)
ax1.scatter([1, 2], [3, 4])
ax1.set_xlim([0, 5])
ax1.set_ylim([0, 5])

ax1 = plt.subplot(131)
ax1.scatter(emp_data_clean["Civilian Labor Force"],emp_data_clean["Unemployment Rate"],emp_data_clean["Year"])
ax1.set_xlim([0.0, 0.5])
ax1.set_ylim([0, 5])

plt.subplot()

plt.scatter(yearly_up_rate)

#10Draw line plot of yearly unemployment rate of US (Year in xaxis and unemployment rate of US in yaxis)

#yearly_up_rate

stck_plot = yearly_up_rate.plot.line()