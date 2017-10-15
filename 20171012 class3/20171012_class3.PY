# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 07:11:44 2017

@author: Vignesh
"""

##############Dictionary#######

math_score_list = [95,67,88,45,84]

math_score_dict_2 = {'bharathwaj':95, 'raj':65, 'kumar':78}
math_score_dict_2['bharathwaj']

math_score_dict = {
        'Bharadwaj':95,
        'Johnson':67,
        'Prabhakaran':88,
        'Roshan':45,
        'Divya':84}
math_score_dict['Johnson']
math_score_dict['Roshan']
# math_score_dict['Kannan'] #throws error
math_score_dict['Kannan'] = 94 # new key value pair is added to the dictionary
del math_score_dict['Kannan']
math_score_dict['Roshan'] = 55 # changing value of key

states = {
    'Oregon':'OR',
    'Florida':'FL',
    'California':'CA',
    'New York': 'NY',
    'Michigan':'MI'}

cities = {
    'CA':'San Fransico',
    'MI': 'Detroit',
    'NY':'Manhattan'}
    
cities['MI'] 
cities['NY']   
cities[states['New York']]

#########numpy array#####
import numpy as np
math_score_list = [95,67,88,45,84]
len(math_score_list)
sum(math_score_list)
# mean
sum(math_score_list)/len(math_score_list)
math_score_array = np.array(math_score_list)
type(math_score_array)
np.mean(math_score_array)
np.median(math_score_array)
np.std(math_score_array)
# numpy commands can also be applied on list
np.mean(math_score_list)
# Conditional slicing
# on list, you have to write a for loop
abv_70 = [math_score_list]
for i in math_score_list:
    if i > 70:
       abv_70.append(i) 
print(abv_70)        

# easy on numpy array
# condition on a numpy array returns a boolean array
condn = math_score_array > 70
# boolean array can be used to slice from a numpy array
# values corresponding to True will get sliced
abv_71 = math_score_array[condn]
abv_71 = math_score_array[math_score_array > 70] # conditional slicing in 1 line
print(abv_71)

# score between 70 to 90
condn1 = math_score_array > 70
condn2 = math_score_array < 90
# & for AND
# | for OR
math_score_array[condn1 & condn2]
# condition on one array can be used to slice another array provided they are of same length
gender = np.array(['M','F','F','M','M'])
condn = gender == "M"
condn = gender == "M"
math_score_array[condn]
condn1 = gender == "M"
condn2 = math_score_array > 80
math_score_array[condn1 | condn2]

## Vectorized Operation
math_score_array = np.array([95,67,88,45,84])
eng_score_array = np.array([78,67,45,39,67])
# element wise sum
math_score_array + eng_score_array
# element wise mean
(math_score_array + eng_score_array)/2



# reshaping a numpy array
array1 = np.array([10,67,84,56,70,25,93,73])
len(array1)
mat1 = array1.reshape(2,4)
mat2 = array1.reshape(4,2)
mat1.T # transpose

# stacking arrays
# LIST of arrays can be stacked
math_score_array = np.array([95,67,88,45,84])
eng_score_array = np.array([78,67,45,39,67])
score_mat1 = np.column_stack([math_score_array,eng_score_array])
score_mat2 = np.row_stack([math_score_array,eng_score_array])

# MATRIX sLICING
# matrix[row_position,column_position]
score_mat1[0,0] # 0th row, 0th column
score_mat1[2,1] # 2nd row, 1st column
score_mat1[:,1] # all rows, 1st column
score_mat1[2,:] #2nd row, all columns
score_mat1[1:4,:] #1st till 3rd row, all columns
score_mat1[[1,3,4],:] # 1st, 3rd and 4th row , all columns

