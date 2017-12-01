# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 07:54:49 2017

@author: Vignesh
"""

import numpy as np
import pandas as pd

#How to get the documentation of the numpy add function from the command line? (★☆☆)
%run `python -c "import numpy; numpy.info(numpy.add)"`

#Create a null vector of size 10 but the fifth value which is 1
Z = np.zeros(10)
Z[4] = 1
print(Z)
#Create a vector with values ranging from 10 to 49
z = np.arange(10,50)
print(z)

#Reverse a vector (first element becomes last)
z1 = np.arange(50)
z1 = z1[::-1]

#Create a 3x3 matrix with values ranging from 0 to 8
z2 = np.arange(9)
z2.reshape(3,3)

# Find indices of non-zero elements from [1,2,0,0,4,0]
nz = np.nonzero([1,2,0,0,4,0])
print(nz)

# Create a 3x3x3 array with random values
z5 = np.random.random((3,3,3))

np.random.random((10,10))
a = []
for i in range(3):
    a.append(int(input()))

s = np.min(a)   
    
min = 0
max = 0
for num in a:
    if num < min &:
        min = num
    elif num > min:
        max = num

print(min)

s = np.min(a)