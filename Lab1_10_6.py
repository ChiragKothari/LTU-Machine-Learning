# Lab 1 exercise 10.6

# Two words are anagrams if you can rearrange the letters from one to spell the other.
# Write a function called is_anagram that takes two strings and returns True if they are anagrams.

from sre_compile import isstring

def is_anagram(s1, s2):
    '''
    Takes two strings and returns True if they are anagrams.
    It is case sensitive, for instance 'House' and 'seuoh' are not anagrams

    Parameters
    --------
    :param s1
        String: first string for comparison
    :param s2
        String: second string fot comparison
    :return
        Boolean: True if s1 and s2 are anagrams, otherwise false
    '''

    if isstring(s1) and isstring(s2):
        print('Provided input are strings!')

        # Convert from string to list of characters
        l1 = list(s1)
        print(l1)
        l2 = list(s2)
        print(l2)

        # Sort the chars in the list in alphabetical order and print sorted lists
        l1.sort()
        print(l1)
        l2.sort()
        print(l2)

        # If sorted list are the same, the two original input strings are anagrams, otherwise they are not
        if l1 == l2:
            print('Provided strings are anagrams!')
            return True
        else:
            print('Provided strings are not anagrams!')
            return False
    else:
        print('Provided input are not strings!')
        return 'Error'


# Test
s1 = 'ringstt'
s2 = 'tstring'

print(is_anagram(s1,s2))
