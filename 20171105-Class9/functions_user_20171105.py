# -*- coding: utf-8 -*-

def print_with_exclamation(word):
    print(word + "!")

print_with_exclamation("Hello") # Function can also be called from inside

def print_sum(x,y):
    print(x+y)

print_sum(5,10)
#print_sum(5) # throws error because both inputs are not passed

# Functions with default values
def print_sum(x = 0, y = 0):
    print(x+y)
print_sum(5,10)
print_sum(5)
print_sum()


# Function which returns values
def return_sum(x = 0, y = 0):
    return(x+y)
op1 = return_sum(5,10)
op2 = return_sum(11,22)
print(op1)
print(op2)

def return_sum(x = 0, y = 0):
    print("Function started") # will be executed
    return(x+y)
    print("Function Completed") # will NOT be executed
op3 = return_sum(5,10)

## SCOPE OF A FUNCTION

def print_val(v):
    v = v+1
    print(v)

print_val(10)
a = 10
print_val(a)
# throws error as v is in the scope of function
# not accessible outside the function
# a variable created in function dies at the end of function
#print(v) 

# Avoid modifying variables in the global environment inside a function
some_list = [3,1,7,5,10]
def impure_function(ip1):
    some_list.append(ip1)
impure_function(20)
print(some_list)
impure_function(30)
print(some_list)

some_list = [3,1,7,5,10]
def pure_function(l1,ip1):
    l1.append(ip1)
    return l1
some_list = pure_function(some_list,20)
print(some_list)


## FUNCTION WITH MULTIPLE RETURN
def max_val(x,y):
    if (x > y):
        return(x)
    elif (y >= x):
        return(y)
max_val(5,10)    
max_val(20,3)

## FUNCTION WHICH RETURNS MULTIPLE VALUES
# Can only be done by wrapping up the outputs to a tuple, list array etc. 
def return_sum_diff(x,y):
    op1 = x + y
    op2 = x - y
    return ([op1,op2])
return_sum_diff(5,10)

## PRACTICE
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

## RECURSION
# A function calling itself
# n! = n * n-1 * n-2 *.....* 1
# 5! = 5*4*3*2*1
def factorial(n):
    if n==1:
        return (1)
    else:
        return (n*factorial(n-1))
 
factorial(5)
factorial(10)







