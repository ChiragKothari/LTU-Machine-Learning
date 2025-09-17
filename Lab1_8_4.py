# Lab 1 exercise 8.4
# The following functions are all intended to check whether a string contains any
#  lowercase letters, but at least some of them are wrong. For each function, describe what the function
#  actually does (assuming that the parameter is a string)

# This function starts with a for loop and evaluates ONLY the first letter of the string, to return boolean True or False depending on
# whether the first letter is lowercase or uppercase respectively.
def any_lowercase1(s):
    for c in s:
        if c.islower():
            return True
        else:
            return False

# This function just returns the string 'True', given that 'c'.islower() is boolean True, and we return the value at beginning
# of first for loop iteration
def any_lowercase2(s):
    for c in s:
        if 'c'.islower():
            return 'True'
        else:
            return 'False'

# This function goes through all the chars in the string with the for loop and for every char checks if it is lowercase
# or uppercase. It returns the evaluation of the last char in the string (can be uppercase or lowercase, we don't know what happened before..)
def any_lowercase3(s):
    for c in s:
        flag = c.islower()
    return flag

# This is the function that evaluates if there are any lowercase letters.
# This function evaluates every char in the string in the for loop. If any of the letters is lowercase, c.islower()
# will become true, and the flag becomes true with the condition flag or c.islower()
# Given that there is an 'or' condition, the flag will always remain true.

def any_lowercase4(s):
    flag = False
    for c in s:
        flag = flag or c.islower()
    return flag

# This function returns False whenever it encounters a capital char in the string. If the string contains a capital letter,
# we will never know if the string contains lowercase letters.
def any_lowercase5(s):
    for c in s:
        if not c.islower():
            return False
    return True


