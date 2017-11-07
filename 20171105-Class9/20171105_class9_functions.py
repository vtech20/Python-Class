# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:12:08 2017

@author: Vignesh
"""

import pandas as pd
import numpy as np
import os

import functions_user as fu
fu.print_with_excl("Hello")

from functions_user import print_sum
print_sum(3,4)
print_sum(5,10)
print_sum(5)
print_sum()

from functions_user import return_sum
op1 = return_sum(5,10)
op2 = return_sum(11,22)
print(op1)
print(op2)

op3 = return_sum(5,10)


from functions_user import print_val
print_val(10)
a = 10
print_val(a)
# throws error as v is in the scope of function
# not accessible outside the function
# a variable created in function dies at the end of function
#print(v) 


from functions_user import impure_function

impure_function(20)
print(some_list)
impure_function(30)
print(some_list)

from functions_user import pure_function
some_list = pure_function(some_list,20)
print(some_list)