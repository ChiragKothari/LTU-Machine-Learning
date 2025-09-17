# Lab 1 exercise 10.7

# Write a function called has_duplicates that takes a list and returns True if there
# is any element that appears more than once. It should not modify the original list.

def has_duplicates(t):
    '''
    Takes a list and returns True if there is any element that appears more than once.
    It does not modify the original list t
    :param t:
        list: input provided
    :return:
        True if the provided list contains duplicates
    '''

    # Create local copy to avoid aliasing
    t_loc = t[:]
    # For every list element, it has to be compared to any other element.
    # Note: list can contain any element, even nested lists.
    # We start with first loop, the second loop has i+1, since it takes the elements after
    # Loop goes from 0 to length -1
    for i in range(len(t_loc)):
        print(i)
        for j in range(i+1, len(t_loc)):
            # When i+1 equals the length, the range is empty. Last string element is not compared with anything left!
            print(t_loc[i])
            print(t_loc[j])
            if t_loc[i] == t_loc[j]:
                # Immediately stops function execution, the == should work for standard Python types
                return True
    return False


    # Smart Alternative:
    # return len(t) != len(set(t))


# Test it
l_test = [[1,2],'q',[2,3], 123]
#l_test = [[1,2],'q',[2,3], 123, [1,2]]
print(has_duplicates(l_test))
