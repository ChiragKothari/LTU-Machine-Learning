# Exercise 17.2. This exercise is a cautionary tale about one of the most common, and difficult to
#  find, errors in Python. Write a definition for a class named Kangaroo with the following methods:
#  1. An__init__ method that initializes an attribute named pouch_contents to an empty list.
#  2. A method named put_in_pouch that takes an object of any type and adds it to
#  pouch_contents.
#  3. A __str__ method that returns a string representation of the Kangaroo object and the con
# tents of the pouch.
#  Test your code by creating two Kangaroo objects, assigning them to variables named kanga and
#  roo, and then adding roo to the contents of kanga’s pouch.

class Kangaroo:
    """
    Class for exercise
    """
    def __init__(self):
        self.pouch_contents = []

    def __str__(self):
        """
        returns string representation of Kangaroo object and the contents of the pouch
        """

        if not self.pouch_contents:
            return "No attributes"

        ret_str = "Kangaroo Object ==> "

        for attr in vars(self):
            ret_str += f"{attr}:{getattr(self, attr)} "

        ret_str += "\nPouch ==> "
        for item in self.pouch_contents:
            for attr_pouch in vars(item):
                # For every attribute in pouch, append the name of the attribute and its value
                ret_str += f"{attr_pouch}:{getattr(item, attr_pouch)} "

        return ret_str

    def put_in_pouch(self,objx):
        """
        takes an object of any type and adds it to pouch_contents
        """
        self.pouch_contents.append(objx)

if __name__ == "__main__":

    # create two Kangaroo objects, assigning them to variables named kanga and roo
    kanga = Kangaroo()
    roo = Kangaroo()


    # add 'roo' to the contents of 'kanga'’s pouch.
    kanga.put_in_pouch(roo)

    # Check
    print(kanga)
    print(roo)

#  Find and fix the bug in BadKangaroo.py
# The default argument value in the __init__ method is mutable (empty list), therefore when initiating the
# class, it refers to an empty list. When kanga and roo are instantiated as objects of the same
# bad class, and then "put_in_pouch" method of kanga is used, the empty list gets filled and that
# empty list is the same to which roo refers to. Therefore, when printing roo, it seems that the method
# "put_in_pouch" was used for roo the same way as it was used for kanga. Solving the issue by replacing
# [] with "None"