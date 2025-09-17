# Lab 1 book Exercise 3.2
# Steps below

# 1. Type this example into a script and test it.
# def do_twice(f):
#     f()
#     f()

# def print_spam():
#     print('spam')

# do_twice(print_spam)

#  2. Modify do_twice so that it takes two arguments, a function object and a value, and calls the
#  function twice, passing the value as an argument.

def do_twice(f, alpha):
    f(alpha)
    f(alpha)

#  3. Copy the definition of print_twice from earlier in this chapter to your script.
def print_twice(bruce):
    print(bruce)
    print(bruce)

#  4. Use the modified version of do_twice to call print_twice twice, passing 'spam' as an argument
do_twice(print_twice,'spam')

#  5. Define a new function called do_four that takes a function object and a value and calls the
#  function four times, passing the value as a parameter. There should be only two statements in
#  the body of this function, not four.

def do_four(g,beta):
    do_twice(g,beta) # first 2 calls
    do_twice(g,beta) # second 2 calls

import math
def test(theta):
    print(math.sin(theta))

do_four(test,(math.pi/2))




