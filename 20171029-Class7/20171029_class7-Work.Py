# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 08:59:08 2017

@author: Vignesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#########Line Plot#############
stockdata = pd.read_csv("E:\\Python Class\\20171029-Class7\\Stock_Price.csv")
stock_plot= stockdata["DELL"].plot.line()
stock_plot.set_xlabel("Time Instance")
stock_plot.set_ylabel("Closing Stock proce of Dell")

stock_plot1= stockdata["Intel"].plot.line()
stock_plot1.set_xlabel("Time Instance")
stock_plot1.set_ylabel("Closing Stock proce of Intel")

#Plotting Multiple columns of a data frame
stock_plot2 = stockdata.plot.line()
stock_plot2.set_xlabel("Time Instance")
stock_plot2.set_ylabel("Closing Stock proce of Intel or Dell")

#if date time is given in index, it becomes a time series data
pd.date_range(start = "2017.01.01",periods=76,freq="M")
pd.date_range(start = "2017.01.01",periods=76,freq="D")

stockdata.index = pd.date_range(start = "2017.01.01",periods=76,freq="D")

plt.scatter("DELL","Intel",data=stockdata)

stock_plot2 = stockdata.plot.line()
stock_plot2.set_xlabel("Time Instance")
stock_plot2.set_ylabel("Closing Stock proce of Intel or Dell")

#################Box Plot##########################
stockdata.loc[:,"DELL"].plot.bar()
stockdata.iloc[0:10,0].plot.bar()
stockdata.iloc[0:10,1].plot.bar()
stockdata.iloc[0:10,:].plot.bar()
#ix is the combination of both loc and iloc features
stockdata.ix[0:10,"DELL"].plot.bar()



##################Control Instruction################

##############If ####################3

if True:
    print("Hello")
    
if False:
    print("Hello")  
    
if 7 > 8:
    print("Condition Satisfied") 
    
if 8 > 7:
    print("Condition Satisfied")  
    
####Nested If ############
if 9 > 7:
   print("Condition 1 Passed") 
   if 5 > 6:
       print("Condition 2 Passed")
####For Loop#######

a =[2,6,1,8,9]
for i in a:
    print(i**2)
    
####### To store it #############
a_squared = []
for i in a:
    a_squared.append(i**2)
print(a_squared)   

#extending a list in loop involves dynamic memory allocation
#Dynamic memory allocation to be avaoided
a_squared = [0.0]*len(a)
pos=0
for i in a:
    a_squared[pos] = i**2
    pos = pos +1
print(a_squared)

#Looping thru Positions for both Slicing and assigning
a_squared = [0.0]*len(a)
for i in range(len(a)):
    a_squared[i] = a[i]**2
print(a_squared)

#Enumeration
#Return a tuple with position and value
for i in enumerate(a):
    print(i)
a_squared = [0.0]*len(a)
for i in enumerate(a):
    val = i[1]
    pos = i[0]
    a_squared[pos] = val**2
print(a_squared)

######List Comprehension########
##Elegant for loops for 1 line implementation
a_squared = [i**2 for i in a]
#or Vectorized operation/numpy
a_squared = np.power(a,2)

####Extracting Odd Numbers######
rnd_no = [13,12,44,65,39,28,80]
odd_numbers = []
for i in rnd_no:
    if i % 2 == 1:
        odd_numbers.append(i)
print(odd_numbers) 
#numpy array
rnd_no_np = np.array(rnd_no)  
odd_numbers = rnd_no_np[rnd_no_np % 2 == 1] 

#List compre
odd_numbers = [i for i in rnd_no if i % 2 == 1]    

#Extracting values greater than 40
#For Loop
greater_40 = []
for i in rnd_no:
    if i > 40:
        greater_40.append(i)
print(greater_40)   
#List Comp
greater_40 = [i for i in rnd_no if i > 40]  
#numpy Vectorized
greater_40 = rnd_no_np[rnd_no_np > 40]   

#################Nested For Loop#############

array1 = np.random.randint(10,100,12)
mat1 = array1.reshape(3,4)

array2 = np.zeros(len(array1))
mat_div_2 = array2.reshape(3,4)

nrow = mat1.shape[0]
ncol = mat1.shape[1]

for i in range(nrow):
    for j in range(ncol):
        mat_div_2[i,j] = mat1[i,j]/2

#Vectorized Operation
mat_div_2 = mat1/2

########################While Loop###############

x = 2
while x < 20:
    print(x**2)
    x=x+1
###Infinite Loop
x = 2
while x < 20:
    print(x**2)
    x=x-1
    
#####Optimization Problems are generally written in a while loop

x = 2
max_iter = 10
iter_count = 1
while x < 20:
      print(x**2)
      x= x - 1
      iter_count = iter_count + 1
      if iter_count >= max_iter:
          break

      
    