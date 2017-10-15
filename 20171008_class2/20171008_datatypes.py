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

#################3 Tuple ########################################
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

