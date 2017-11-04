# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 07:49:13 2017

@author: Vignesh
"""
import numpy as np

#1. Create a list of birth years of 5 friends/family member (e.g: br_yr = [1986, 1989, 1975, 1981, 1978]). Calculate their age (years alone) as of 2017 using 3 approaches and save it a list. Assume their birth days are over in 2017 
#Regular for loops 
#List comprehension
#Vectorized operation using numpy array

#Regular for loops
br_yr = [1986, 1989, 1975, 1981, 1978]
age_lst = [0.0]*len(br_yr) 
for i in range(len(br_yr)):
    age_lst[i] = 2017 - br_yr[i]
print(age_lst)

#List Comprehen
age_lst1 = [2017-i for i in br_yr]
print(age_lst1)

#Vectorized Operation
br_yr_np = np.array(br_yr)
age_lst2 = 2017 - br_yr_np
print(age_lst2)

#Create a string “this is a python exercise which is neither too easy nor too hard to be solved 
#in the given amount of time”. Split the string to list of individual words [Hint: split command. 
#Don’t search in classwork]. 
#Remove words like ‘is’, ‘a’ and ‘the’ programmatically using 3 approaches 

a = "this is a python exercise which is neither too easy nor too hard to be solved in the given amount of time"
spt_lst = a.split(" ")
#for Loop
final_lst = []
for i in spt_lst:
    if i != "is" and i != "a" and i != "the":
        final_lst.append(i)
print(final_lst)

#List Comprehen
final_lst1 = [i for i in spt_lst if i != "is" and i != "a" and i != "the"]
print(final_lst1)

#Vectorized Operation
spt_lst_np = np.array(spt_lst)
final_lst2 = spt_lst_np[(spt_lst_np != "is") & (spt_lst_np != "a") & (spt_lst_np != "the")]
print(final_lst2)