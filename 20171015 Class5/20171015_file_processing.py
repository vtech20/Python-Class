# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import datetime
# Use front slash (/) or double back slash (\\) to read the file
acs_2013_data = pd.read_csv("C:\\Karthik\\Learning\\Python\\Green\\data\\ACS_13_5YR_S1903.csv")
acs_2008_data = pd.read_csv("C:\\Karthik\\Learning\\Python\\Green\\data\\ACS_08_3YR_S1903.csv")

# It is not recommended to provide full path
# Have a project folder, maintain the same folder structure
# Set the working directory to the project folder
# RELATIVE INDEXING
os.getcwd()
os.chdir("C:\\Karthik\\Learning\\Python\\Green")
os.getcwd()

acs_2013_data = pd.read_csv("data\\ACS_13_5YR_S1903.csv")
acs_2008_data = pd.read_csv("data\\ACS_08_3YR_S1903.csv")

# Pandas while detecting data type could do mistakes
# Pass columns names and associated data type as a dictionary
acs_2013_data.dtypes
acs_2013_data = pd.read_csv("data\\ACS_13_5YR_S1903.csv",
                            dtype = {'GEO.id2':str})
acs_2013_data.dtypes
acs_2008_data = pd.read_csv("data\\ACS_08_3YR_S1903.csv",
                            dtype = {'GEO.id2':str})
# Chane all columns to one data type
#acs_2008_data = pd.read_csv("data\\ACS_08_3YR_S1903.csv",
#                            dtype = object)

## Try following questions for ACS 2013 data
# Q1. slice the first 7 columns
acs_2013_s = acs_2013_data.iloc[:,0:7]

# Q2. rename the column names as follows
#["ID","FIPS","State",
#                    "Total Household", "Total Household MOE",
#                    "Income","Income MOE"]
acs_2013_s.columns = ["ID","FIPS","State",
                    "Total Household", "Total Household MOE",
                    "Income","Income MOE"]
# Q3. calculate average income of US
acs_2013_s["Income"].mean() # pandas mean function
np.mean(acs_2013_s["Income"]) # numpy mean function
# Q4. what is the maximum income and which state is that?
max_income = np.max(acs_2013_s["Income"])
condn = acs_2013_s["Income"] == max_income
acs_2013_s.loc[condn,"State"]
# Q5. what is the minimum income and which state is that?
min_income = np.min(acs_2013_s["Income"])
condn = acs_2013_s["Income"] == min_income
acs_2013_s.loc[condn,"State"]
# Q6. get the list of states which are above average in household income
avg_income = np.mean(acs_2013_s["Income"])
condn = acs_2013_s["Income"] > avg_income
acs_2013_s.loc[condn,"State"]
# Q7. get the income of texas state
condn = acs_2013_s["State"] == "Texas"
acs_2013_s.loc[condn,"Income"]
# Q8. what is the state which has the 2nd highest income
# Implementation 1A
acs_2013_s_income_sorted = acs_2013_s.sort_values("Income",ascending = False)
acs_2013_s_income_sorted.iloc[1,2]

# Implementation 1B
#acs_2013_s_income_sorted.iloc[1,"State"] # throws error because iloc cannot be used to slice by index
acs_2013_s_income_sorted.loc[1,"State"] # returns Alaska because Alaska has index 1
# reset index will make index to start from 0 and copied previous index as a new column
acs_2013_s_income_sorted2 = acs_2013_s_income_sorted.reset_index()
# to avoid new column
acs_2013_s_income_sorted2 = acs_2013_s_income_sorted.reset_index(drop = True)
acs_2013_s_income_sorted2.loc[1,"State"]
#acs_2013_s_income_sorted3 = acs_2013_s_income_sorted.reset_index()
#del acs_2013_s_income_sorted3["index"]
#acs_2013_s_income_sorted4 = acs_2013_s_income_sorted3.reset_index(drop = True)

# Implementation 2
max_income = acs_2013_s["Income"].max()
second_highest_income = acs_2013_s.loc[acs_2013_s["Income"] != max_income,"Income"].max()
acs_2013_s.loc[acs_2013_s["Income"] == second_highest_income,"State"]

# Implementation 3
sorted_income = np.sort(acs_2013_s["Income"])
second_highest_income = sorted_income[-2]
acs_2013_s.loc[acs_2013_s["Income"] == second_highest_income,"State"]

######### DATA MERGING ####################################33
# equivalent to JOIN operation in SQL
# Extracting only FIPS code, Total households and Income 
income_2013 = acs_2013_data.iloc[:,[1,3,5]]
income_2013.columns = ["FIPS","Tot_Household_2013","Income_2013"]
income_2008 = acs_2008_data.iloc[:,[1,3,5]]
income_2008.columns = ["FIPS","Tot_Household_2008","Income_2008"]

# if reference column name is same in both the tables
merged_income = pd.merge(income_2008,income_2013,on="FIPS")

# if reference column names are different
income_2013.columns = ["FIPS_2013","Tot_Household_2013","Income_2013"]
income_2008.columns = ["FIPS_2008","Tot_Household_2008","Income_2008"]
merged_income = pd.merge(income_2008,income_2013,
                         left_on="FIPS_2008", right_on="FIPS_2013")

# JOINING OPTIONS
# Adding artificial states to the tables
income_2008.loc[52,:] = ["88",12345,123]
income_2013.loc[52,:] = ["99",54321,543]
# FIPS code 88 present in 2008 but not in 2013
# FIPS code 99 present in 2013 but not in 2008

# INNER JOIN
merged_income_inner = pd.merge(income_2008,income_2013,how = "inner",
                         left_on="FIPS_2008", right_on="FIPS_2013")
# OUTER JOIN
merged_income_outer = pd.merge(income_2008,income_2013,how = "outer",
                         left_on="FIPS_2008", right_on="FIPS_2013")
# LEFT JOIN
merged_income_left = pd.merge(income_2008,income_2013,how = "left",
                         left_on="FIPS_2008", right_on="FIPS_2013")
merged_income_left["Tot_Household_2008"] = merged_income_left["Tot_Household_2008"].astype(int)
# RIGHT JOIN
merged_income_right = pd.merge(income_2008,income_2013,how = "right",
                         left_on="FIPS_2008", right_on="FIPS_2013")

########## FILE WRITING #################################
merged_income.to_csv("output/merged_income.csv")

#Q: Appending current date to file name automatically
now = datetime.datetime.now().date()
fname = "output/"+str(now)+"_merged_income.csv"
merged_income.to_csv(fname,index = False)












