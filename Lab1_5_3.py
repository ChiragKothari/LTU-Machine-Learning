# Lab 1 exercise 5.3

#  1. Write a function named is_triangle that takes three integers as arguments, and that prints
#  either “Yes” or “No”, depending on whether you can or cannot form a triangle from sticks
#  with the given lengths.

def is_triangle(a,b,c):
    # If any of the three lengths is greater than the sum of the other two, then you cannot
    #  form a triangle. Otherwise, you can. (If the sum of two lengths equals the third, they
    #  form what is called a “degenerate” triangle.)

    if(a>b+c) or (b>a+c) or (c>a+b):
        print('No')
    else:
        print('Yes')

#  2. Write a function that prompts the user to input three stick lengths, converts them to integers,
#  and uses is_triangle to check whether sticks with the given lengths can form a triangle.

def ask_triangle():
    # list comprehension, requesting user to input sticks' lengths 3 times and converting input to int, with for constructor
    # It was discovered to be an efficient way of retrieving values in python with a single command and a formatted string.
    stick_lengths = [int(input(f"Enter the length of stick number {i+1}: ")) for i in range(3)]
    # pass values directly to check the function
    is_triangle(stick_lengths[0],stick_lengths[1],stick_lengths[2])

# Test
ask_triangle()


