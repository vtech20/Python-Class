# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 08:51:13 2017

@author: Vignesh
"""

import numpy as np
import pandas as pd
import os


#1 Write a function which accepts a list/array/series as input and returns 
#the difference between mean and median
other_list = []
some_list = [3,1,7,5,10]
def pure_function(l1,ip1):
    l1 = np.mean(ip1) - np.median(ip1)
    return l1
s = pure_function(other_list,some_list)

#np.mean(some_list)
#np.median(some_list)

#2 Write a function (max_var2_corresponding) which accepts a data frame (df) as input 
#along with 2 column names (var1, var2) in the data frame. 
#Calculate the maximum value in var1 column of df. 
##Return the value of var2 corresponding to maximum value of var1 

def max_var2_corresponding(df1,var1,var2):
    s1 = df1.loc[df1[var1] == np.max(df1[var1]),var2]
    return(s1)
#a. Test Case 1:     
math_score_array = np.array([95,67,88,45,84]) 
eng_score_array = np.array([78,67,45,39,67]) 
gender_array = np.array(["M","M","F","M","F"]) 
score_df = pd.DataFrame({         
            'Maths':math_score_array,         
            'English':eng_score_array,         
            'Gender':gender_array}) 
score_df.index = ["R1001","R1002","R1003","R1004","R1005"]

max_var2_corresponding(score_df,"Maths","English") 

max_var2_corresponding(score_df,"English","Gender")

#b. Test Case 2: 
    
emp_details_dict = { 
        'Age': [25,32,28], 
        'Income': [1000,1600,1400] 
        } 
emp_details = pd.DataFrame(emp_details_dict) 
emp_details.index = ['Ram','Raj','Ravi'] 

max_var2_corresponding(emp_details,"Income","Age")   