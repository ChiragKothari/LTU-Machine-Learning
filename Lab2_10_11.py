# Two words are a “reverse pair” if each is the reverse of the other. Write a program
# that finds all the reverse pairs in the word list.

# The word list is a list of strings, according to https://github.com/AllenDowney/ThinkPython2/blob/master/code/inlist.py

def find_reverse(word_list, set_original_list):
    """
    Function to find all reverse pairs in a list of words (a list of strings).
    No assumptions made on the input list of string
    Case sensitive function "Ex" and "xe" are not returned as reverse pairs
    :param word_list:
        list : list of strings
    :param set_original_list:
        bool : if True, remove duplicates in original word_list
    :return:
        list: paired_reverse, a list with the found reverse pairs
    """
    if set_original_list:
        # There might be duplicates in the original list provided as input, which we might do not want to have
        word_list = list(set(word_list))

    # Compute length of the list
    len_wl = len(word_list)

    # Initialization of empty list to populate with pairs of reverse words, if existing
    paired_reverse = []
    # Reverse all the words in the provided list at once with list comprehension
    word_list_reversed = [word_to_reverse[::-1] for word_to_reverse in word_list] # by default start:stop:step, with -1 reverse order
    print(word_list_reversed)

    # Compare every word in the original list with every word in the list of reverse words, starting from following word
    # If we find a reverse pair, we move directly to inspect the next word
    for i, word_actual in enumerate(word_list):
        found = False
        j = i+1
        # If we find a word that is equal to any of the reverse words, append them in the pair for final display and move to next word
        while (found == False) and (j < len_wl):
            if word_actual == word_list_reversed[j]:
                # If we find a reverse pair, store it in the output list and mark it found
                paired_reverse.append([word_actual,word_list[j]])
                found = True
            j+=1
    return paired_reverse




if __name__ == '__main__':

    # test_word_list = ["example", "ex", "xe", "xe", "Xe", "elpmaxe"]
    test_word_list = ["example", "ex", "xe", "xe", "Xe", "elpmaxe", "ex"] # with duplicates

    # set_original_list == False will try to find for every string in the list if there is at least a reverse pair
    # set_original_list == True starts by removing possible duplicates in the provided list
    paired_words = find_reverse(test_word_list, set_original_list = False)
    print(paired_words)


# Note: book solution can be found at https://github.com/AllenDowney/ThinkPython2/blob/master/code/reverse_pair.py


