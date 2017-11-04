# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 08:21:02 2017

@author: Vignesh
"""

import pandas as pd
import numpy as np
import os
mtcars_data = pd.read_csv("E:\\Python Class\\Assignments\\mtcars.csv")

#Data Frame properties and quality check
#Q3
print("Number of Rows : ",mtcars_data.shape[0])
#Q4
print("Number of Columns : ",mtcars_data.shape[1])
#Q5
print("Column Names are: ",mtcars_data.columns)
#Q6
print("Statistical Summary",mtcars_data.describe())

#Data Frame Slicing
#Q7
print("Average MPG of all Cars",np.mean(mtcars_data['mpg']))
#Q8
print("Average MPG of Auto Trans Cars",
np.mean(mtcars_data.loc[mtcars_data["am"] ==0,"mpg"]))
#Q9
print("Average MPG of Manual Trans Cars",
np.mean(mtcars_data.loc[mtcars_data["am"] ==1,"mpg"]))
#Q10
print("Average Disp of 4 Gear Cars",
np.mean(mtcars_data.loc[mtcars_data["gear"] ==4,"disp"]))
#Q11
print("Average Horse power of cars with 3 carb",
np.mean(mtcars_data.loc[mtcars_data["carb"] ==3,"hp"]))
#Q12
print("Average mpg of automatic cars with 4 gears: ",
np.mean(mtcars_data.loc[(mtcars_data["gear"] ==4) & (mtcars_data["am"] ==0) ,"mpg"]))
#Q13
# Average qsec of cars with mpg above average mpg and weight below average weight 
np.mean(mtcars_data.loc[(mtcars_data["mpg"] > np.mean(mtcars_data["mpg"])) & 
(mtcars_data["wt"] < np.mean(mtcars_data["wt"])) ,"qsec"])
#Q14
print("Entire row of the vehicle which has the highest miles per gallon: ",
mtcars_data.loc[mtcars_data["mpg"] == mtcars_data["mpg"].max(),:])

#or
mtcars_data.iloc[mtcars_data["mpg"].idxmax(),:]
#Q15
# Entire row of vehicle with the highest horsepower 
highest_hp = mtcars_data.loc[mtcars_data["hp"] == mtcars_data["hp"].max(),:]
print(highest_hp)
#Q16
#Mileage and hp of car with highest weight 
highest_wt = mtcars_data.loc[mtcars_data["wt"] == mtcars_data["wt"].max(),["mpg","hp"]]
print(highest_wt)
#Q17
d_ratio = mtcars_data["mpg"] / mtcars_data["carb"]
d_ratio.mean()
#Q18
#Weight of the car with the minimum displacement 
min_disp = mtcars_data.loc[mtcars_data["disp"] == mtcars_data["disp"].min(),"wt"]
print(min_disp)

#Q19
# Slice all columns of 3 gear cars 
print("All Cols of 3 Gear Cars: ",mtcars_data.loc[mtcars_data["gear"] ==3,:])
#Q20
print(mtcars_data.loc[mtcars_data["am"] ==1,["mpg","disp","hp"]])
#Q21
#What is average mpg of 3, 4 and 5 gear cars. Save output as a list/array/series 
avgmpgseries = pd.Series()
for x in mtcars_data['gear'].unique():
    mpgdata = (mtcars_data['mpg'][mtcars_data['gear'] == x]).mean()
    avgmpgseries.set_value(x,mpgdata)
    print(avgmpgseries)

#Q22
#What is average hp, average wt, average qsec, average vs for 3, 4 and 5 gear cars. 
#Save output as a matrix or data frame    
avghpseries = pd.Series()
avgqsecseries = pd.Series()
avgwtseries = pd.Series()
avgvsseries = pd.Series()
for x in mtcars_data['gear'].unique():
    hpdata = (mtcars_data['hp'][mtcars_data['gear'] == x]).mean() 
    qsecdata = (mtcars_data['qsec'][mtcars_data['gear'] == x]).mean() 
    wtdata = (mtcars_data['wt'][mtcars_data['gear'] == x]).mean()
    vsdata = (mtcars_data['vs'][mtcars_data['gear'] == x]).mean() 

    avghpseries.set_value(x,hpdata)
    avgqsecseries.set_value(x,qsecdata)
    avgwtseries.set_value(x,wtdata)
    avgvsseries.set_value(x,vsdata)

    d_avg = {"Avg Hp": avghpseries,
              "Avg Qsec":avgqsecseries,
              "Avg Wt":avgwtseries,
              "Avg VS":avgvsseries}
              
    df_all_avg = pd.DataFrame(d_avg)
    print(df_all_avg)