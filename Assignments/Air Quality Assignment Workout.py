# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 18:08:32 2017

@author: Vignesh
"""

import pandas as pd
import numpy as np
import os

airquality_data = pd.read_csv("E:\\Python Class\\Assignments\\airquality.csv")


#Data Frame Properties and Quality Check
#3.
airquality_data.shape[0]
#4.
airquality_data.shape[1]
#5.
airquality_data.columns
#6.
s = airquality_data[airquality_data['Ozone'].isnull()]
s.shape[0]

#or
print(airquality_data['Ozone'].isnull().sum())
#
#Data Frame Slicing 
#7.
s1=airquality_data['Solar.R'].isnull()
condx=s1 ==False
airquality_data.loc[condx,:]

#or
solarNotNull = airquality_data[np.isfinite(airquality_data['Solar.R'])]

print(solarNotNull)

#8.
airquality_data['Ozone'].mean()

#9.
s2 = np.mean(airquality_data['Temp'])
condn1 = airquality_data['Temp'] > s2
final_val = airquality_data.loc[condn1,'Solar.R']
final_avg = np.mean(final_val)

#0r
avg_temp  = np.mean(airquality_data['Temp'])

cond = airquality_data['Temp'] >avg_temp

print(cond)

daysGreat = airquality_data.loc[cond ,["Day","Solar.R"]]

print(daysGreat)

print('average value of Solar.R on days with temperature above average temperature: ' , np.mean(daysGreat['Solar.R']))

#10
condn2 = airquality_data['Day'] == 15
f_list = airquality_data.loc[condn2,:]

#or

#11
condn3 = airquality_data['Month'] == 6
condn4 = airquality_data['Month'] == 8
f_list1 = airquality_data.loc[condn3 | condn4,:]

#12
s3 = airquality_data['Solar.R'].mean()
condn5 = airquality_data['Solar.R'] > s3
f_list2 = airquality_data.loc[condn5 & condn1,'Ozone'].mean()

#For Loops
length = len(airquality_data.index)

ozoneSum = 0

solarSum = 0

windSum = 0

tempSum = 0

print(airquality_data)

airquality_data['Ozone'].fillna(0, inplace=True)

print(airquality_data)

airquality_data['Solar.R'].fillna(0, inplace=True)

print(airquality_data)

airquality_data['Wind'].fillna(0, inplace=True)

print(airquality_data)

airquality_data['Temp'].fillna(0, inplace=True)

print(airquality_data)

for index, row in airquality_data.iterrows():

            ozoneSum+=row['Ozone']

            solarSum+=row['Solar.R']

            windSum +=row['Wind']

            tempSum +=row['Temp']

# Excel does not take into account the values with nan

allAvg = pd.Series()

allAvg.set_value('Ozone',ozoneSum / length)

allAvg.set_value('Solar.R',solarSum / length)

allAvg.set_value('Wind',windSum / length)

allAvg.set_value('Temp',tempSum / length)

print(allAvg)

#print('Ozone Avg: ',ozoneSum / length)

#print('Solar Avg: ',solarSum / length)

#print('Wind Avg: ',windSum / length)

#print('Temp Avg: ',tempSum / length)



# Q 14  Calculate month-wise average Ozone and save in a list/array/series 

# without for loop

monthlyGrouped = airquality_data.groupby(['Month']).mean()

ozoneMonthlyData = pd.Series(monthlyGrouped['Ozone'])

print(ozoneMonthlyData)



# with for loop

ozoneSeries = pd.Series()

for x in airquality_data['Month'].unique():

    ozoneData = (airquality_data['Ozone'][airquality_data['Month'] == x]).mean()

    #print(ozoneData)

    ozoneSeries.set_value(x,ozoneData)

print(ozoneSeries)

     

# Q15. Calculate month-wise average Ozone, Solar, Wind and Temperature and save in a matrix/data 

ozoneAvg = pd.Series()

solarAvg = pd.Series()

windAvg = pd.Series()

tempAvg = pd.Series()



for x in airquality_data['Month'].unique():

    ozoneData = (airquality_data['Ozone'][airquality_data['Month'] == x]).mean()

    solarData= (airquality_data['Solar.R'][airquality_data['Month'] == x]).mean()

    windData = (airquality_data['Wind'][airquality_data['Month'] == x]).mean()

    tempData = (airquality_data['Temp'][airquality_data['Month'] == x]).mean()

    

    ozoneAvg.set_value(x,ozoneData)

    solarAvg.set_value(x,solarData)

    windAvg.set_value(x,windData)

    tempAvg.set_value(x,tempData)

    

    

    data = {'ozoneAvg': ozoneAvg,

            'solarAvg': solarAvg,

            'windAvg': windAvg,

            'tempAvg': tempAvg

            }

avgsData = pd.DataFrame(data)

print(avgsData)

    

#    print(ozoneData) 
        