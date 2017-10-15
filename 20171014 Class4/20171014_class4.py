# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 19:34:21 2017

@author: Vignesh
"""

import pandas as pd

###############Pandas Data Frame#################

# Creating Data Frame from Dictionary 

dict1 = {
         'çol1':[1,2,3],
         'col2':[3,4,5]}
df_dict1 = pd.DataFrame(dict1)

# rows are mapped based on Index

dict2 = {'col1':pd.Series([1,2,3], index=['a','b','c']),
         'col2':pd.Series([3,4,5], index=['b','a','c'])}
df_dict2 = pd.DataFrame(dict2)

dict3 = {'col1':pd.Series([1,2,3], index=['a','b','c']),
         'col2':pd.Series([3,4,5], index=['b','a','d'])}
df_dict3 = pd.DataFrame(dict3)

dict1 = {
         'çol1':[1,2,3],
         'col2':[3,4,5]}
df_dict1 = pd.DataFrame(dict1)
df_dict1.index=['a','b','c']

#df from dictionary
import numpy as np
math_score_array = np.array([95,67,88,45,84])
eng_score_array = np.array([78,67,45,39,67])
gender_array = np.array(['M','M','F','F','M'])
score_dict = {
              'Maths':math_score_array,
              'English':eng_score_array,
              'Gender':gender_array}
score_df = pd.DataFrame(score_dict)
score_df.index=['R1001','R1002','R1003','R1004','R1005']  

######### Attributes of a Dataframe
score_df.index
score_df.columns
score_df.shape
score_df.shape[0] # no of rows
score_df.shape[1] #no of columns

######## Data Frame Slicing################
score_df["Maths"]
math_score_sliced = score_df["Maths"]
type(math_score_sliced) # Every column in a data frame is a series

#Slicing more than 1 columns
col = ["Maths","English"]
selected_columns = score_df[col]
selected_columns = score_df[["Maths","English"]]
type(selected_columns)

#Slicing a row
#score_df["R1004"] # throws Error

#Slice like a matrix in [row,col] format
#slicing using index and column names
score_df.loc["R1004","Maths"]
score_df.loc["R1004",:]
score_df.loc[:,"Maths"]
roi = ["R1004","R1001"]
coi = ["Maths","Gender"]
score_df.loc[roi,coi]
score_df.loc[["R1004","R1001"],["Maths","Gender"]]

#Slicing by position
score_df.iloc[3,2]
score_df.iloc[3,:]
score_df.iloc[:,2]
roi=[3,1]
coi=[0,2]
score_df.iloc[roi,coi]
score_df.iloc[0:4,:]

#conditional Boolean slicing
#.loc for Boolean slicing

#maths score of male students
condn = score_df["Gender"] == 'M'
score_df.loc[condn,"Maths"]
score_df.loc[condn,["Maths","Gender"]]
score_df.loc[condn,:]

#Q2, all columns of students who scored above 70 in Maths


#q2.
condn1 = score_df["English"] > 60
mathsscore = score_df.loc[condn1,"Maths"]
np.mean(mathsscore) 

#q3

avg_math = np.mean(score_df["Maths"])     
condn = score_df["Maths"] > avg_math
