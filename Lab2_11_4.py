# If you did Exercise 10.7, you already have a function named has_duplicates that
#  takes a list as a parameter and returns True if there is any object that appears more than once in the list.
#  Use a dictionary to write a faster, simpler version of has_duplicates.
#  Solution: https://thinkpython.com/code/has_duplicates.py.

def has_duplicates_dict(list_input,hashable):
    """
    Takes a list as a parameter and returns True if there is any object that appears more than once in the list
    Uses a dictionary to do that, exploiting the fact that dictionary keys must be unique
    Input:
        list_input : list
            generic list of values
        hashable : bool
            to tell if the list is hashable or not
    Output:
        bool: True if there are duplicates, False if there are not
    """
    if not hashable:
        list_input = [str(elm) for elm in list_input]

    # Compute a dictionary from a list by storing key and corresponding indexes.
    # Python understands dictionary is representation of key:value pairs and will not allow duplicates
    dict_dupl = {key:idx for idx,key in enumerate(list_input)} # this will work only if the elements of the list are hashable
    # One can see by debugging that python is keeping the latest duplicate value
    if len(list_input) == len(dict_dupl):
        return False
    else:
        return True


# Solution from the book below, more elegant but does not work with unashable list
def has_duplicates(t):
    """Checks whether any element appears more than once in a sequence.

    Simple version using a for loop.

    t: sequence
    """
    d = {}
    for x in t:
        if x in d: # as soon as a key is identified in the dictionary, it means we have a duplicated value therefore we return a True
            return True
        d[x] = True # This does not work if list is unhashable !!
    return False # If no duplicates are found, after looping through all the elements we can return a False



if __name__ == "__main__":

    # Test the functions to learn more by practice
    # Three list are given as input, one unhashable with duplicates and two list of strings (hashable)
    # Comment and decomment input to test. Change hashable input in has_duplicates_dict to True if list is hashable
    # If unhashable, conversion is done in the function

    list_in = [123,"apple",[1,'2'],123] # non hashable list --> raises error [str(elm) for elm in list_in]
    #list_in = ["no","apple","3","no"] # hashable list --> Returns True.
    #list_in = ["apple", "3", "no"]  # hashable list --> Returns False

    # Written function
    print(has_duplicates_dict(list_in, hashable = False))

    # Test book function. This has problem with unhashable list though
    #print(has_duplicates(list_in))



