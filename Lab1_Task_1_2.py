# Implement several functions from scratch without using built-in functions. These functions
# will operate on the input list of grades and return specific values. Here are the functions you need to
# implement:

def Min(t):
    '''
    Find and return the minimum value from the input list
    :param t:
        list
    :return:
    '''

    if not t:
        print("The list is empty")
        return None

    Min = t[0]
    for elm in t[1:]:
        if elm < Min:
            Min = elm
    return Min

def Max(t):
    '''
    Find and return the maximum value from the input list
    :param t:
        list
    :return:
        maximum value from input list
    '''

    if not t:
        print("The list is empty")
        return None

    Max = t[0]
    for elm in t[1:]:
        if elm > Max:
            Max = elm
    return Max

def Mean(t):
    '''
    Calculate and return the average value across values in the input list
    :param t:
        list
    :return:
        average value across values from input list
    '''

    if not t:
        print("The list is empty")
        return None

    avg = t[0]
    c = 1 #counter, cannot use len builtin function

    for elm in t[1:]:
        avg += elm
        c += 1

    return avg/c


def Variance(t):
    '''
    This function should compute and return the spread of the values in the input list, which
    measures how far each number in the set is from the mean
    :param t:
        list
    :return:
        variance of values from input list
    '''

    if not t:
        print("The list is empty")
        return None

    # Compute Mean of the values to compute squared differences
    avg = Mean(t)

    # Compute squared differences with list comprehension
    sqr_diff = [(s-avg)**2 for s in t]

    sum_sqr_diff = 0
    for se in sqr_diff:
        sum_sqr_diff += se


    #Compute Variance as the mean of squared differences
    #return Mean(sqr_diff) <--- Chirag
    return sum_sqr_diff/(len(t)-1)

def Stdv(t):
    '''
    Implement this function to calculate and return the dispersion of the input list
    relative to its mean
    :param t:
        list
    :return:
        standard deviation of values from input list
    '''

    if not t:
        print("The list is empty")
        return None

    # Compute Mean of the values to compute squared differences
    var = Variance(t)

    # Compute Standard deviation as square root of Variance
    return var ** 0.5

def Median(t):
    '''
    sort the input list to return the middle number from the sorted list as the median value
    :param t:
        list
    :return:
        median value from the input list t
    '''

    if not t:
        print("The list is empty")
        return None

    # Calculate length of list t
    len_t = 0
    len_t_odd = False # length zero is even, length 1 is odd
    for _ in t:
        len_t += 1
        # At the end of the loop we will also know if the length is odd or even
        len_t_odd = not len_t_odd

    # Sort the values in list t_sort
    i = 0
    t_sort = t[:]

    while i < len_t:
        j = 0
        while j < len_t - i - 1:
            # If j-th element is greater than the next one, swap it
            if t_sort[j] > t_sort[j+1]:
                t_sort[j], t_sort[j+1] = t_sort[j+1], t_sort[j]
            # increment j
            j += 1
        # increment i
        i += 1

    # Compute Median as middle number of sorted values, consider if the length of list is odd or even
    if len_t_odd:
        idx_odd = int((len_t-1)/2) # casting into int datatype
        return float(t_sort[idx_odd])
    else:
        return (float(t_sort[int(len_t/2)] + t_sort[int(len_t/2 - 1)]))/2

def MedianAbsDev(t):
    '''
    Compute the average distance of the input points from the median and return result
    :param t:
        list
    :return:
        median absolute deviation value from the input list t, that is the median of the absolute deviations from the data's median
    '''

    if not t:
        print("The list is empty")
        return None

    # Start from the median
    med = Median(t)

    # Compute absolute deviations with list comprehension
    med_abs_dev = [(s - med) if s >= med else (med - s) for s in t]

    # Compute median absolute deviation as median of deviations from the median
    return Median(med_abs_dev)


# Task 2 get some insight
# Continue by defining a list. Imagine that you are a teacher and you just received the grades
# that a group of students got in an exam. This list will be used to calculate some interesting values.

grades = [8,6,1,7,8,9,8,7,10,7,6,9,7]


# 2.1 Obtain: Min, max, mean, standard deviation, median and median absolute deviation.
# 2.2 Explain the results obtained

print(f'Grades: {grades}')
print(f'Minimum: {Min(grades)}') # minimum value in the list
print(f'Maximum: {Max(grades)}') # maximum value in the list
print(f'Mean: {Mean(grades)}') # average value in the list of grades, affected by outlier
print(f'Standard deviation: {Stdv(grades)}') # how much the data are distributed around the mean value, it is more sensitive to outliers (e.g. 1) with respect to median absolute deviation
print(f'Median: {Median(grades)}') # the value for which half data are higher and half data lower, usually most frequent value that appears in a set of values [1,6,6,7,7,7,7,8,8,8,9,9,10]
print(f'Median Absolute Deviation: {MedianAbsDev(grades)}') # used to understand deviations from the median and more stable to outliers (e.g. 1), ordered absolute deviations around median, median is 7 [6,1,1,0,0,0,0,1,1,1,2,2,3]. Median of these values is 1

