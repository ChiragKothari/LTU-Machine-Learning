# Write a function called draw_rect that takes a Turtle object and a Rectangle and
# uses the Turtle to draw the Rectangle.
# Write a function called draw_circle that takes a Turtle and a Circle and draws the Circle.

import turtle
import math

class Rectangle:
    """ Class to represent a rectangle
    Attributes: width, height, corner"""

class Point:
    """Class to represent a point in space
    Attributes: x, y"""

class Circle:
    """Class to represent a circle in space
    Attributes: center, radius"""


def draw_rect(t,r):
    """
    Takes a Turtle object and a Rectangle and uses the Turtle to draw the Rectangle
    Input:
        t : Turtle
            Turtle object (instance) used to draw the rectangle
        r : Rectangle
            Rectangle object (instance) representing the rectangle to draw
    """

    # Call Turtle instance method "pen up" to avoid drawing when positioning the lower left corner
    t.pu()
    t.fd(r.corner.x)
    t.lt(90)
    t.fd(r.corner.y)
    # After positioning the lower left corner, place the "pen down"
    t.pd()
    # Now, we are ready to draw
    for operations in range(2):
        t.fd(r.height)
        t.rt(90)
        t.fd(r.width)
        t.rt(90)

    # Open the window to draw
    turtle.mainloop()

def draw_circle(t, c):
    """
    Takes a Turtle and a Circle and draws the Circle
    Input:
        t : Turtle
            Turtle object (instance) used to draw the rectangle
        r : Circle
            Circle object (instance) representing the circle to draw
    """
    # Retrieve value of attribute radius from circle
    r = c.radius
    circumference = 2 * math.pi * r
    n = 100
    length_edge = circumference/n
    angle = 360 / n

    for i in range(n):
        t.fd(length_edge)
        t.lt(angle)

    # Open the window to draw
    turtle.mainloop()

if __name__ == "__main__":

    # Create instance of class Rectangle
    rect = Rectangle()
    # Customize the rectangle do draw
    # Add attributes to the instance and assign them values, corner is the object Point()
    rect.width = 100
    rect.height = 200
    rect.corner = Point()
    rect.corner.x = 100 # lower left corner x coordinate (in pixel)
    rect.corner.y = 50 # lower left corner y coordinate (in pixel)


    # Create instance of class Circle
    circ = Circle()
    # Customize circle to draw
    # Add attributes to the instance and assign them values, center is the object Point()
    circ.radius = 75
    circ.center = Point()
    circ.center.x = 150  # center x coordinate (in pixel)s
    circ.center.y = 100  # center y coordinate (in pixel)

    # Create Turtle object
    bob = turtle.Turtle()

    # Call draw_rect and provide rectangle object and turtle object to draw the rectangle in the window
    #draw_rect(bob, rect)

    # Call draw_rect and provide rectangle object and turtle object to draw the rectangle in the window
    draw_circle(bob, circ)



