# -*- coding: utf-8 -*-

import numpy as np

################ IF #########################################################
if True:
    print("Hello") # will be executed

if False:
    print("Hello") # will NOT be executed

if 7 > 8:
    print("condition satisfied")

if 8 > 7:
    print("\ncondition satisfied")
    
### Nested if
if 9 > 7:
    print("\ncondition 1 passed")
    if 5 > 6:
        print("\nCondition 2 passed")

### else, elif (else if)
x = 5
if x == 4:
    print ("entered if")
elif x == 5:
    print("entered else if")
else:
    print("entered else")

## Multiple conditions and, or
x = 1000
if (x >= 5  and x <= 100):
     print ("within range")
else:
    print("out of range")
     

    
################## FOR LOOP ##################################################

# C Style
#for (int i = 0; i<10;i++){
##use iterating variable to do repetitive tasks
#}

#for i in list/array/series:
#    # use i for repetitive tasks

a = [2,6,1,8,9]
for i in a:
    # iteration 1: i = 2, 4
    # iteration 2: i = 6, 36
    print(i**2)

a_squared=[]
for i in a:
    a_squared.append(i**2)
print(a_squared)

# extending a list in loop involves dynamic memory allocation
# dynamix memory allocation to be avoided as much as possible
## looping through the values; maintaining separate variable for position
a_squared = [0.0]*len(a)
pos = 0
for i in a: 
    a_squared[pos] = i**2
    pos = pos+1
print(a_squared)

## looping through positions, use it for both slicing and assigning
a_squared = [0.0]*len(a)
for i in range(len(a)): 
    a_squared[i] = a[i]**2
print(a_squared)

## ENUMERATION
# Return a tuple with position and value
for i in enumerate(a):
    print(i)    
    
a_squared = [0.0]*len(a)
for i in enumerate(a):
    val = i[1]
    pos = i[0]
    a_squared[pos] = val**2
print(a_squared)

## LIST COMPREHENSION
# Elegant for loops for 1 line implementations
a_squared = [i**2 for i in a]

## VECTORIZED OPERATION/NUMPY OPERATION
a_squared = np.power(a,2)

### Extracting Odd Numbers
rnd_no = [13,12,44,65,39,28,80]
# for loop
odd_numbers = []
for i in rnd_no:
    if i % 2 == 1:
        odd_numbers.append(i)
print(odd_numbers)
# list comprehension
odd_numbers = [i for i in rnd_no if i % 2 == 1]
print(odd_numbers)
# numpy array
rnd_no_np = np.array(rnd_no)
odd_numbers = rnd_no_np[rnd_no_np % 2 == 1]

### Extracting values above 40
# for loop
abv_40 = []
for i in rnd_no:
    if i > 40:
        abv_40.append(i)
print(abv_40)
# list comprehension
abv_40 = [i for i in rnd_no if i > 40]
# numpy vectrorized operation
abv_40 = rnd_no_np[rnd_no_np > 40]

############ NESTED FOR LOOP ##############################

array1 = np.random.randint(10,100,12)
mat1 = array1.reshape(3,4)

array2 = np.zeros(len(array1))
mat_div_2 = array2.reshape(3,4)

nrow = mat1.shape[0]
ncol = mat1.shape[1]

for i in range(nrow): # looping through the rows
    for j in range(ncol): # looping through the columns for each row
        mat_div_2[i,j] = mat1[i,j]/2

## VECTORIZED OPERATION
mat_div_2 = mat1/2

################# WHILE LOOP #################################################
# C Style
#while(condition){
# # instructions runs repeatedly till condition fails
#}

x = 2
while x < 20:
    print(x**2)
    x = x+1

# Infinite loop
#while True:
#    print("Python")

# Infinite Loop
x = 2
while x < 20:
    print(x**2)
    x = x-1

# Optimization problems are  generally written inside a while loop
# All optimization problems will have a breaking condition

########## BREAK ##############################################
x = 2
max_iter = 1000
iter_count = 1
while x < 20:
    print(x**2)
    ##### Optimization logic which is not converging ##########
    x = x-1
    ####### End of Optimization logic###############################    
    iter_count = iter_count+1
    if iter_count >= max_iter:
        break # Terminate the loop

## find whether a number is prime or not
n = 11
for i in range(2,round(n/2)):
    if n % i == 0:
        print ("not a prime")
        break # I don't need to test further
    
############## Continue #######################################
salary = np.random.randint(20000,400000,300)
salary_with_bonus = salary.copy()
for i in range(len(salary)):
    if salary[i] > 100000:
        continue # remaining lines will NOT be executed
    salary_with_bonus[i] = salary[i]*1.1
print(salary_with_bonus)

































































