# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 18:21:48 2017

@author: Vignesh
"""

#1 (a)
l1 = list(range(21))
print(l1)
l2 = list(range(1,21))
print(l2)

#1 (b)
l3 = list(range(20,0,-1))
print(l3)

#1 (c)
l4 = list(range(1,21))
l5 = list(range(19,0,-1))
l6 = l4 + l5
print(l6)

#or 
l7 = list(range(1,21)) + list(range(19,0,-1))
print(l7)

#1 (d)
tmp = [4,6,3]

#1 (e)
l8 = tmp * 10
print(l8)

#1 (f)

l9 = l8 + [4]
print(l9)

#1 (g)
l10 = [4]*10 
l11 = [6]*20
l12 = [3]*30
l14 = l10 + l11 + l12
print(l14)