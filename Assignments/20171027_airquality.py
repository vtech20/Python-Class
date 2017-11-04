# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:54:52 2017

@author: karthik.ragunathan
"""

import pandas as pd
import numpy as np

airquality = pd.read_csv("data/airquality.csv")

#3. How many rows in the data?
airquality.shape[0]
#4. How many columns in the data?
airquality.shape[1]
#5. What are the column names?
airquality.columns


"""
6. How many null values in Ozone column 
(Note: nans are treated as nulls. 
There is a pandas function to catch nulls)
"""
airquality["Ozone"].isnull().sum()

"""
7. Slice from airquality a dataframe which only has 
rows with valid entries for Solar.R. 
Remove rows which has null values in Solar.R column"""

aq_solarna_removed = airquality.loc[airquality["Solar.R"].notnull(),:]

# removing rows which has NULL in any of the row
aq_clean = airquality.loc[airquality["Solar.R"].notnull() &
               airquality["Ozone"].notnull() &
               airquality["Wind"].notnull() &
               airquality["Temp"].notnull(),:]
aq_clean = airquality.dropna() # simplistic

#8. What is the average value of Ozone column?
airquality["Ozone"].mean()
np.mean(airquality["Ozone"])

#9. What is the average value of Solar.R on 
# days with temperature above average temperature?
avg_temp = airquality["Temp"].mean()
airquality.loc[airquality["Temp"] > avg_temp,"Solar.R"].mean()
#10. Slice only records of 15th day of each month
airquality.loc[airquality["Day"] == 15,:]
#11. Slice records of 6th and 8th month alone
condn1 = airquality["Month"] == 6
condn2 = airquality["Month"] == 8
airquality.loc[condn1 | condn2,:]
"""
12. What is the average ozone values of the days where 
both Solar.R and Temperature are above their averages?
"""
condn1 = airquality["Solar.R"] > airquality["Solar.R"].mean()
condn2 = airquality["Temp"] > airquality["Temp"].mean()
airquality.loc[condn1 & condn2,"Ozone"].mean()

#13. Calculate average values of Ozone, Solar, Wind and Temperature
avg_columns = [
airquality["Ozone"].mean(),
airquality["Solar.R"].mean(),
airquality["Wind"].mean(),
airquality["Temp"].mean()]
avg_columns = pd.Series(avg_columns,index = ["Ozone","Solar.R","Wind","Temp"])

#14. Calculate month-wise average Ozone
month_wise_avg_ozone = [
airquality.loc[airquality["Month"] == 5,"Ozone"].mean(),
airquality.loc[airquality["Month"] == 6,"Ozone"].mean(),
airquality.loc[airquality["Month"] == 7,"Ozone"].mean(),
airquality.loc[airquality["Month"] == 8,"Ozone"].mean(),
airquality.loc[airquality["Month"] == 9,"Ozone"].mean()]
month_wise_avg_ozone = pd.Series(month_wise_avg_ozone,index = range(5,10))

#15. Calculate month-wise average Ozone, Solar, Wind and Temperature. 
month_wise_avg_ozone = [
airquality.loc[airquality["Month"]== 5,"Ozone"].mean(),
airquality.loc[airquality["Month"]== 6,"Ozone"].mean(),
airquality.loc[airquality["Month"]== 7,"Ozone"].mean(),
airquality.loc[airquality["Month"]== 8,"Ozone"].mean(),
airquality.loc[airquality["Month"]== 9,"Ozone"].mean()]
month_wise_avg_solar = [
airquality.loc[airquality["Month"]== 5,"Solar.R"].mean(),
airquality.loc[airquality["Month"]== 6,"Solar.R"].mean(),
airquality.loc[airquality["Month"]== 7,"Solar.R"].mean(),
airquality.loc[airquality["Month"]== 8,"Solar.R"].mean(),
airquality.loc[airquality["Month"]== 9,"Solar.R"].mean()]
month_wise_avg_wind = [ 
airquality.loc[airquality["Month"]== 5,"Wind"].mean(),
airquality.loc[airquality["Month"]== 6,"Wind"].mean(),
airquality.loc[airquality["Month"]== 7,"Wind"].mean(),
airquality.loc[airquality["Month"]== 8,"Wind"].mean(),
airquality.loc[airquality["Month"]== 9,"Wind"].mean()]
month_wise_avg_temp = [
airquality.loc[airquality["Month"]== 5,"Temp"].mean(),
airquality.loc[airquality["Month"]== 6,"Temp"].mean(),
airquality.loc[airquality["Month"]== 7,"Temp"].mean(),
airquality.loc[airquality["Month"]== 8,"Temp"].mean(),
airquality.loc[airquality["Month"]== 9,"Temp"].mean()]
month_wise_avg_values = np.column_stack([month_wise_avg_ozone,
                                        month_wise_avg_solar,
                                        month_wise_avg_wind,
                                        month_wise_avg_temp])
month_wise_avg_values = pd.DataFrame(month_wise_avg_values)
month_wise_avg_values.columns = ["Ozone","Solar.R","Wind","Temp"]
month_wise_avg_values.index = range(5,10)