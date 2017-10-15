# -*- coding: utf-8 -*-
### PYTHON IS CASE SENSITIVE ###################

################ integer ######################
# any number without decimal space is assigned as integer
a = 2
type(a)
c = -10
type(c)

################float ####################
# any number with decimal values is assigned as float
b = 123.45673456
type(b)

############ String ##################
# any value within quotes is a string
s = "god"
type(s)
s1 = 'god' # single quotes can also be used
type(s)
s2 = "999" # even a number within quotes is a string
type(s2)
# s2 + 5 # throws error
int(s2) + 5
#int(s) # throws error as god could not be converted as an integer
s3 = "g" # even single character is a string
type(s3)

str1 = "rama"
str2 = "ravana"
concat_str = str1 + " killed " + str2
print(concat_str)
print(len(str1)) # number of characters in a string
print(len(str2))

################### Boolean #########################
f = True
type(f)
#g = true # Python is case sensitive
h = False
H = True
# output of conditions are boolean
5 > 6
6 > 5
5 == 6 # == for equality check
5 == 5
5 != 5
6 != 5
not(6 == 5)
"god" == "god"
"god"=="ogd"
"god" == "God"
"god" == 'god'

# more than one condition could be executed using 'and', 'or' operators
(6 < 5) and (8 > 7)
(6 > 5) and (7 > 8)
(6 > 5) or (7 > 8)
not(6 < 5) and (8 > 7)
not((6 < 5) and (8 > 7))

################ Complex #########################################
h = 5 + 10j
type(h)
m = 5 - 10j
h*m
abs(h*m)

### type casting
int(10.5)
float(3)
int("999")
int("05")
#int("god") throws error

############################################################################

################# Tuple ########################################
tup1 = (1,2,3,4,5)
type(tup1)
tup2 = ("rama","ravana","sita","lakshmana")
# mix of data types are allowed in an array
tup3 = (1,2,"god",5.67,True,6+10j)

# Typle slicing by position
# [] for slicing
# position starts with 0
tup1[0]
tup1[1]

# tuples are immutable
 # tup1[2] = 100 #throws error
 
########################### List ##################################3
l1 = [1,2,3,4,5]
type(l1)
l1[2] = 100 # values in a list can be changed
l3 = [1,2,"god",5.67,True,6+10j]

# concatenate 2 lists using +
l4 = l1 + l3
print(l4)

# appending a value
l1.append(100)
print(l1)

# data simulation
l4 = list(range(100))
len(l4)
l5 = list(range(1,101))
l6 = list(range(1,101,2))
l8 = list(range(100,9,-2))

five_rep_100 = [5]*100
print(five_rep_100)
one_to_five_rep_20 = list(range(1,6))*20
print(one_to_five_rep_20)

## List slicing
l8[0]
l8[len(l8) - 1]

l8[-1] # negative indexing
l8[-2] # last but one

l8[5:11] # 5th to 10th position
l8[5:] #5th to last
l8[:6] #0th to 5th position
l8[1:-1] #1st to last but one
l8[0:2] + l8[-2:] # beginning 2 and ending 2

l8[::2] # extracting even positions
l8[1::2] # extracting odd positions

# List searching
l10 = [10,67,84,56,70,25,93]
67 in l10
68 in l10

word_list = ['egg','milk','cheese','butter']
'milk' in word_list
'meat' in word_list
'meat' not in word_list

del l10[0:2]
print(l10)

################ Dictionary ###############################################
# Key-Value pairs within curly braces
# Keys should be of data types which are immutable
# Keys could be string, integer, float, tuple
# Value could of any data type


math_score_list = [95,67,88,45,84]
type(math_score_list)
print(math_score_list[2])

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

# You can have list of dictionaries
# You can have dictionaries of list
student_dict = {
        'Bharadwaj':[95,True,'M'],
        'Johnson':[67,False,'M'],
        'Prabhakaran':[88,True,'M'],
        'Roshan':[45,True,'M'],
        'Divya':[84,False,'F']}

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

# Add a city 'Orlando' to 'FL'
cities['FL'] = 'Orlando'
cities['FL'] = ['Orlando','Miami']

# what is the name of the city in state "MI"
cities['MI']
# what is the city name in 'New York' State
cities[states['New York']]


############### numpy array ###############################################
import numpy as np
math_score_list = [95,67,88,45,84]
len(math_score_list)
sum(math_score_list)
# mean
sum(math_score_list)/len(math_score_list)
# median
# you have to sort the values, extract the middle position

math_score_array = np.array(math_score_list)
type(math_score_array)
np.mean(math_score_array)
np.median(math_score_array)
np.std(math_score_array)

# numpy commands can also be applied on list
np.mean(math_score_list)

# Conditional slicing
# on list, you have to write a for loop
#abv_70 = []
#for i in math_score_list:
#    if i > 70:
#       abv_70.append(i) 
#print(abv_70)
# for loop will be covered later


# easy on numpy array
# condition on a numpy array returns a boolean array
condn = math_score_array > 70
# boolean array can be used to slice from a numpy array
# values corresponding to True will get sliced
abv_70 = math_score_array[condn]
abv_70 = math_score_array[math_score_array > 70] # conditional slicing in 1 line

# score between 70 to 90
condn1 = math_score_array > 70
condn2 = math_score_array < 90
# & for AND
# | for OR
math_score_array[condn1 & condn2]

# condition on one array can be used to slice another array provided they are of same length
gender = np.array(['M','F','F','M','M'])
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

############### numpy matrix ##############################################

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

############## pandas Series ##################################################
import pandas as pd
# numpy array can also be sliced by position
# no option to slice by a unique identifier

math_score_series = pd.Series([95,67,88,45,84])
type(math_score_series)
math_score_series = pd.Series([95,67,88,45,84], 
                              index = ['Bharadwaj','Johnson','Prabhakaran','Roshan','Divya'])

math_score_series[2]
math_score_series['Prabhakaran']

# all numpy functions can be applied on pandas series
np.mean(math_score_series)
np.std(math_score_series)

# conditional slicing
math_score_series[math_score_series > 70]

# vectorized operations can also be done
math_score_series/2

# convert dictionary to pandas Series
math_score_dict = {
        'Bharadwaj':95,
        'Johnson':67,
        'Prabhakaran':88,
        'Roshan':45,
        'Divya':84}
math_score_series = pd.Series(math_score_dict)

math_score_dict_1 = {
        95:'Bharath',
        89:'Vignesh'}
        
math_score_series1 = pd.Series(math_score_dict_1)

states = {
    'Oregon':'OR',
    'Florida':'FL',
    'California':'CA',
    'New York': 'NY',
    'Michigan':'MI'}
states_series = pd.Series(states)















