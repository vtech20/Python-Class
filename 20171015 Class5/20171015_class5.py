# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 09:15:12 2017

@author: Vignesh
"""

import pandas as pd
import numpy as np
import os

#use front slash or double back slash to open the file
acs_2008_data = pd.read_csv("E:\\Python Class\\20171015 Class5\\ACS_08_3YR_S1903.csv")
acs_2013_data = pd.read_csv("E:\\Python Class\\20171015 Class5\\ACS_13_5YR_S1903.csv")

os.getcwd() # Current Personal Directory
os.chdir("E:\\Python Class\\20171015 Class5\\") # Change Current Working Direc
os.getcwd()

acs_2008_data = pd.read_csv("ACS_08_3YR_S1903.csv")
acs_2013_data = pd.read_csv("ACS_13_5YR_S1903.csv")

#q1
c = acs_2013_data.iloc[:,0:7]
type(c)
#q2
c.columns = ["ID","FIPS","State","Total Household","Total Household NOE", "Income", "Income NOE"]

#q3
us_avg = np.mean(c["Income"])
#q4
max_income = np.max(c["Income"]) 
condn12 = c["Income"] == max_income
cc_state = c.loc[condn12,"State"]

#q5
min_income = np.min(c["Income"])
condn13 = c["Income"] == min_income
cc_state_min = c.loc[condn13,"State"]

#Q6
avg_household = np.mean(c["Income"])
condn14 = c["Income"] > avg_household
cc_state_abavg = c.loc[condn14,"State"]

#Q7
condn15 = c["State"] == "Texas"
cc_state_texes = c.loc[condn15,"Income"]

#q8
sechigh = c.sort_values("Income",ascending=False)
sechigh.iloc[1,2]

sechigh1 = np.sort(c["Income"])
cond16 = c["Income"] == sechigh1[-2]
statesec = c.loc[cond16,"State"]

sechigh2 = c.reset_index(drop = True)
sechigh2.loc[1,"State"]


#Select and ctrl1 for batch of comment

#sdsf
#dfggdf
#hfghffgj

acs_2013_data.dtypes
acs_2013_data = pd.read_csv("ACS_13_5YR_S1903.csv",dtype={'GEO.id2':str})
acs_2013_data.dtypes
acs_2008_data = pd.read_csv("ACS_08_3YR_S1903.csv",dtype={'GEO.id2':str})

####### Data Merging ###########
#Equi to Join
income_2013 = acs_2013_data.iloc[:,[1,3,5]]
income_2013.columns = ["FIPS","Tot_Household_2013","Income 2013"]

income_2008 = acs_2008_data.iloc[:,[1,3,5]]
income_2008.columns = ["FIPS","Tot_Household_2008","Income 2008"]

# if the reference column is same in both the tables
merged_income = pd.merge(income_2008,income_2013,on="FIPS")

#if the reference column is different in both the tables
income_2013.columns = ["FIPS_2013","Tot_Household_2013","Income 2013"]
income_2008.columns = ["FIPS_2008","Tot_Household_2008","Income 2008"]
merged_income = pd.merge(income_2008,income_2013,left_on="FIPS_2008",right_on="FIPS_2013")

#Joining Options
#Adding artificial states to the tables
income_2008.loc[52,:] = ["88",12345,123]
income_2013.loc[52,:] = ["99",54321,123]

#Inner Join
merged_income_inner = pd.merge(income_2008,income_2013,how="inner",
                               left_on="FIPS_2008",right_on="FIPS_2013")



#Outer Join


merged_income_outer = pd.merge(income_2008,income_2013,how="outer",
                               left_on="FIPS_2008",right_on="FIPS_2013")

#Left Join

merged_income_left = pd.merge(income_2008,income_2013,how="left",
                               left_on="FIPS_2008",right_on="FIPS_2013")

#right Join
merged_income_right = pd.merge(income_2008,income_2013,how="right",
                               left_on="FIPS_2008",right_on="FIPS_2013")


############File Writing###########

merged_income.to_csv("merged_income.csv")
merged_income.to_csv("merged_income.csv",index = False)

import datetime
now = datetime.datetime.now().date()
fname = str(now) + "_merged_incove.csv"
merged_income.to_csv(fname,index=False)

