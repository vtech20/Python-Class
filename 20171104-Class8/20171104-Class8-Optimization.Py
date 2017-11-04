# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 09:13:30 2017

@author: Vignesh
"""

import pandas as pd
import numpy as np


df = pd.DataFrame({
'A':['foo','bar','foo','bar','foo','bar','foo','foo'],
'B':['one','two','two','one','one','two','two','one'],
'C':np.random.rand(8),
'D':np.random.rand(8),
'E':np.random.rand(8),
'F':np.random.rand(8),
'G':np.random.rand(8)})



coi = df.columns[2:]
avg_cols = pd.Series([0.0]*len(coi),index = coi)
for i in coi:
    avg_cols[i] = np.mean(df[i])
print(avg_cols)

nrows = df.shape[0]
avg_rows = pd.Series([0.0]*nrows)
for i in range(nrows):
   avg_rows[i] = np.mean(df.iloc[i,2:])
print(avg_rows)   
   
#######################Apply####################################
#Axis =0 for Col operation
#axis = 1 for row operation

df.iloc[:,2:].apply(np.mean, axis=1)
df.iloc[:,2:].apply(np.mean, axis=0)

df.iloc[:,2:].mean()

df.iloc[:,2:].apply(np.median, axis=1)
df.iloc[:,2:].apply(np.median, axis=0)

df_foo = df.loc[df["A"]=="foo",:]
df_bar = df.loc[df["A"]=="bar",:]

df_grouped_A = df.groupby("A")
df_foo = df_grouped_A.get_group("foo")
df_bar = df_grouped_A.get_group("bar")
for i in df_grouped_A:
    print (i)
    
df_grouped_B = df.groupby("B")
for i in df_grouped_B:
    print (i)

df_grouped_AB = df.groupby(["A","B"])
for i in df_grouped_AB:
    print (i)       

######### Group by and apply,aggregate ###############################

# repeating a function on a column for each group/segment
df_grouped_A["C"].apply(np.mean)
df_grouped_A["D"].apply(np.mean)
df_grouped_A["G"].apply(np.mean)

# repeating a function on multiple columns of each group
df_grouped_A[["C","D","E","F","G"]].apply(np.mean)
df_grouped_A[["C","D","E","F","G"]].apply(np.sum)

# repeating a function on multiple columns of each group
df_grouped_A[["C","D","E","F","G"]].agg(np.mean)
df_grouped_A[["C","D","E","F","G"]].agg(np.median)
df_grouped_A[["C","D","E","F","G"]].agg(np.sum)

# repeating multiple functions on multiple columns of each group
df_grouped_A[["C","D","E","F","G"]].agg([np.mean,np.median])

df_grouped_A[["C","D","E"]].agg({
        "C":np.mean,
        "D":np.median,
        "E":[np.min,np.max]})

airquality = pd.read_csv("E:\\Python Class\\Assignments\\airquality.csv")  
#Calculate average values of Ozone, Solar, Wind and Temperature 
airquality[["Ozone","Solar.R","Wind","Temp"]].apply(np.mean)
#
monthlyGrouped = airquality.groupby(['Month']).mean()

ozoneMonthlyData = pd.Series(monthlyGrouped['Ozone'])

print(ozoneMonthlyData)
#
#Calculate month-wise average Ozone, Solar, Wind and Temperature.
monthlyGrouped = airquality.groupby(['Month'])
monthlyGrouped[["Ozone","Solar.R","Wind","Temp"]].agg(np.mean)