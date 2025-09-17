#  Write a function called sed that takes as arguments a pattern string, a replacement
#  string, and two filenames; it should read the first file and write the contents into the second file
#  (creating it if necessary). If the pattern string appears anywhere in the file, it should be replaced
#  with the replacement string.

#  If an error occurs while opening, reading, writing or closing files, your program should catch the
#  exception, print an error message, and exit.

def sed(pattern,replacement,filename1,filename2):
    """
    Try-except blocks are placed around single operation, alternative is to encapsulate entire function corpse into a
    try-except block and print captured exception for programmer to debug
    Input:
        pattern : string
            string containing pattern
        replacement : string
            string containing a replacement
        filename1 : path
            string of filepath for file that has to be read
        filename2 : path
            string of filepath for file to write into
    """

    # Open the first file in read mode, read content and close file.
    # Print exception and return if there are issues in opening the file in read mode
    try:
        file1 = open(filename1,'r')
        file1_content = file1.read()
        # Close the file after content is read
        file1.close()
    except Exception as e:
        print(f'Error generated in reading file1 block:\n{e}')
        return

    # Open the second file in write mode, print exception and return if there are issues in opening the file in write mode
    # File is created if it does not exist
    try:
        file2 = open(filename2, 'w')
    except Exception as e:
        print(f'Error generated in opening file2 block:\n{e}')
        return

    # Replace content to write
    content_to_write = file1_content.replace(pattern,replacement)

    # Write desired content in second file and close it
    try:
        file2.write(content_to_write)
        file2.close()
    except Exception as e:
        print(f'Error generated in writing content into file2 block:\n{e}')
        return


if __name__ == "__main__":

    # Test function
    pattern = "AH!IA "
    replacement = "__"
    filename1 = "file1_to_read.txt"
    filename2 = "file2_to_write.txt"
    sed(pattern, replacement, filename1, filename2)





