# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:14:08 2017

@author: Vignesh
"""

def print_with_excl(word):
    print(word + "!")

def print_sum(x,y):
    print(x+y)    
    
def print_sum(x=0,y=0):
    print(x+y)   
    
def return_sum(x=0,y=0):
    return(x+y)      
    
def return_sum(x=0,y=0):
    print("Function Started")
    return(x+y)      
    print("Function Ended")
    
## SCOPE OF A FUNCTION

def print_val(v):
    v = v+1
    print(v)



# Avoid accessing variables in the global environment inside a function
some_list = [3,1,7,5,10]
def impure_function(ip1):
    some_list.append(ip1)


def pure_function(l1,ip1):
    l1.append(ip1)
    return l1
    
## PRACTICE
def calculate(x,y,option):
    if (option == "sum"):
        return(x+y)
    elif (option == "diff"):
        return(x-y)
    elif (option == "mult"):
        return(x*y)
    else:
        return(x/y)
    ####

op1 = calculate(5,3,"sum") # 8
op2 = calculate(5,3,"diff") # 2
op4 = calculate(5,3,"mult") # 15
op6 = calculate(6,3,"div") #2



def calculate(x=0,y=0,options="sum"):
    if(options=="sum"):
        return (x+y)
    elif(options =="diff"):
        return (x-y)
    elif(options=="mult"):
        return (x*y)
    elif(options=="div"):
        return (x/y)
    else:
        raise Warning("Unknown option")

## POSITIONAL MATCHING
op1 = calculate(5,3,"sum") # 8
op2 = calculate(5,3,"diff") # 2
op3 = calculate(5,3,"mult") # 15
op4 = calculate(6,3,"div") #2
op5 = calculate(6,3,"junk")

## ARGUMENT MATCHING
calculate(y = 3, x = 5, options = "diff")
calculate(options = "diff", y = 3, x = 5)
# throws error. positional matching cannot follow argument matching
# calculate(options = "diff", 3, 5)
# argument matching can follow position matching
calculate(3, 5, options = "diff")
calculate(3,options="sum")

########################################3
###### LAMBDA FUNCTIONS 
# One line compact functions

def polynomial(x):
    return x**2 + 5*x + 5
polynomial(5)
polynomial(11)

polynomial_lambda = lambda x:x**2+5*x + 5
polynomial_lambda(5)
polynomial_lambda(11)
# More than one input in lambda
polynomial_lambda2 = lambda x,y:x**2+5*y + 5
polynomial_lambda2(5,10)
polynomial_lambda2(2,45)

###### FUNCTIONAL PROGRAMMING
# Passing function as an argument to another function
def add_five(x):
    return (x+5)
add_five(10)

def add_ten(x):
    return (x+10)

def call_function(func,x):
    return func(x)
call_function(add_five,10)

def apply_twice(func,x):
    return(func(func(x)))
apply_twice(add_five,10)
apply_twice(add_ten,10)

def apply_more(func,x):
    return(func(func(func(x))))
    
apply_more(add_five,10)    

    