# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 07:33:25 2017

@author: Vignesh
"""

import pandas as pd
import numpy as np
import random as rd
l1 = np.array([5]*25)
l2 = np.array(list(range(25,0,-1)))
l3 = np.array(list(range(0,50,2)))
l4 = np.array(rd.sample(range(10,100),25))
d_dict = {'A':l4,
          'B':l1,
          'C':l2,
          'D':l3}
df_dict1 = pd.DataFrame(d_dict)
df_dict1.index = [list(range(1001,1026))]

#Q1
s = pd.Series(df_dict1['A'])
#Q2
df2 = df_dict1[['A','C']]
#Q3
df3 = df_dict1.iloc[:,[0,2]]
#Q4
s[0:5]
#Q5
df4 = df_dict1.iloc[2:19,:]
#Q6
aa = df_dict1['A'].median()
cond33 = df_dict1['A'] > aa
df5 = df_dict1.loc[cond33,:]
