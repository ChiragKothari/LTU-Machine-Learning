# Write a program that reads a file, breaks each line into words, strips whitespace and
# punctuation from the words, and converts them to lowercase.
# Hint: The string module provides a string named whitespace, which contains space, tab, new
# line, etc., and punctuation which contains the punctuation characters.
# Also, you might consider using the string methods strip, replace and translate.

# Import modules
import string
punctuation = string.punctuation

# strip method removes leading and trailing whitespaces at beginning and end of the string

# A .txt file is created to test functionalities.

if __name__ == '__main__':

    filename = "test_file_13_1.txt"
    # Read the file into lines
    f = open(filename,"r")
    # generate list to collect lines
    lines = []
    # Loop through lines
    for line in f:
        # Break each line into words. We assume words are between punctuation and spaces
        for ch in punctuation:
            # Strip whitespace and punctuation
            line = line.replace(ch," ")
            line = line.strip()
            # Convert to lowercase
            line = line.lower()
        lines.append(line)

    print(lines)
    # Close the file
    f.close()

