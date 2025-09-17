import skimage
from skimage.color import rgb2gray
import os
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
from skimage import io

# Task 1
print('### Task 1 <-----------------------------')
# Load the image data
filename_path = os.getcwd()
#filename = os.path.join(filename_path,'coins.jpg')
filename = os.path.join(filename_path,'astronaut.jpg')
image = io.imread(filename)

print(image.shape) # coins/astronaut is grayscale 2D data/RGB color image (303,384)/(512, 512, 3)
print(image.dtype) # coins/astronaut uint 8/uint8
# coins
print(image[1,100]) # pixel value at position 1,100
# astronaut
#print(image[1,100,1]) # We print the G component in the RGB channel at position 1,100

# Visualize the image with either the “io.imshow” or the “plt.imshow” commands
io.imshow(image)
io.show()
#ax = plt.imshow(image)
#plt.show()

# Task 2: Color space conversion
print('### Task 2 <-----------------------------')
grayscale = rgb2gray(image) # the input array must have size 3 along `channel_axis`, got (303, 384)
# • Convert the “coins” and “astronaut” images to grayscale.
# • Explain what happens with the dtype and size of each image before and after the conversion.
# coins is already 2D (not RGB), we get error
# astronaut gets converted, the RGB channels are transformed into grayscale value, the pixel value range is [0,1]
#io.imshow(grayscale)
#io.show()

# Task 3: Image rescale and resize
print('### Task 3 <-----------------------------')
print('RESCALE')
for scale in [0.25, 0.50, 0.75]:
    image_rescaled = rescale(grayscale, scale, anti_aliasing=False) # rescale of given factor 'scale', preserves aspect ratio
    print(f'Image shape after rescale factor {scale}: {image_rescaled.shape}')
    io.imshow(image_rescaled)
    io.show()

print('RESIZE')
# Resize allows to specify output image shape instead of a scaling factor
resizing_factor1 = 4
resizing_factor2 = 7
image_resized=resize(grayscale, (image.shape[0] // resizing_factor1, image.shape[1]// resizing_factor2),anti_aliasing=True) # Resize to specified values without necessarily preserving aspect ratio
print(f'Image shape after resize: {image_resized.shape}')
io.imshow(image_resized)
io.show()


# Task 4: Image thresholding
# Use the grayscale converted “coins” image from Task 2, calculate the histogram and select a threshold to
# distinguish the object from the background. Based on the histogram, select a value for the variable t and
# replace the ?? on line 19 with your selected value. Observe the data type of the grayscale image and
# select the t value accordingly (e.g uint8 or float64).

print('### Task 4 <-----------------------------')
import numpy as np
from skimage.exposure import histogram
from skimage.util import img_as_ubyte

filename_path = os.getcwd()
filename = os.path.join(filename_path,'coins.jpg')
image = io.imread(filename)
io.imshow(image)
io.show()

print(f'Task 4 Image datatype: {image.dtype}')

ax = plt.hist(image.ravel(), bins=256) # histogram, plot pixel intensity values distribution and get rid off background by
# Thresholding understanding the background is characterized by low pixel intensity values and coins are high intensity values

t = np.uint8(125)
binary = image < t # Here we get binary image

fig, ax = plt.subplots()
plt.imshow(binary, cmap="gray")
plt.show()

# Task 5: Template matching
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.feature import match_template

# Retrieve from the library standardized image object
image = data.coins() # image from skimage library

# Extract image portion as a template
coin = image[170:220, 75:130]

# Image matching, match the template with image content
result = match_template(image, coin)
# We find the index of location of maximum of the FLATTENED 1D image, so we need to convert the index in the 1D array back to
# a pair of indexes as in the 2D image, np.unravel_index helps doing it. Naturally, it demands knowledge of image shape, output space
ij = np.unravel_index(np.argmax(result), result.shape)
# From row, column to column, row, useful for plotting the rectangle and the small circle
x, y = ij[::-1]

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

# This is the coin, the template
ax1.imshow(coin, cmap=plt.cm.gray)
ax1.set_axis_off()
ax1.set_title('template')

# This is the full image
ax2.imshow(image, cmap=plt.cm.gray)
ax2.set_axis_off()
ax2.set_title('image')

# highlight template coin with rectangle
hcoin, wcoin = coin.shape # Extract the template size
rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none') # Create red rectangle to display around the template coin
ax2.add_patch(rect) # Adding the rectangles to full image plotted above

ax3.imshow(result)
ax3.set_axis_off()
ax3.set_title('`match_template`\nresult')
# highlight matched region
ax3.autoscale(False)
ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

plt.show()

# Task 6: Own implementation of template matching algorithm

# We require 2 crucial segments.
# Source image (I): The picture inside which we are hoping to
# find out a match to the template image.
# Template image (T): The patch image that can be compared
# to the template image and our objective is to discover the
# most effective technique for the best matching region.
# Matching technique not solely takes the similarity measure
# however it computes the inaccuracy among images reckoning
# on the difference by means of Mean Squared Error(MSE)
# metric.
# wherever, Temporary (u, v) and Target (u, v) are the strength
# values of Template  image and the input image respectively
# 1.1.3 Template-based approach
# Template-based template matching approach could probably
# require sampling of a huge quantity of points, it‟s possible to
# cut back the amount of sampling points via diminshing the
# resolution of the search and template images via the same
# factor and performs operation on resulting downsized images
# (multiresolution,or Pyramid (image processing)) , providing a
# search window of data points inside the search image in order
# that the template doesn‟t have to be compelled to look for
# each possible data point and the mixture of the two.

def template_matching(original_img, portion):
    """
    Perform template matching on original_img for the template extracted in rectangle defined by portion
    Choice is to pick the minimum of the absolute difference between template and sub-image
    Algorithm implementation from An Overview of Various Template Matching Methodologies in Image Processing
    Paridhi Swaroop
    M.Tech(Computer Science)
    Banasthali Vidhyapeeth

    Rajasthan
    Input
        original_img : 2D or 3D image
            RGB or grayscale image
        portion: list
            contains 4 numbers to identify rows and columns of rectangular template to extract from provided image
    """

    # Show original img
    io.imshow(original_img)
    io.show()

    # If RGB, convert to grayscale
    if len(original_img.shape) == 3:
        # Color space conversion from RGB to gray
        original_img = rgb2gray(original_img)


    # Using Template Image
    # Extract image portion as a template:
    # Template image may be a small portion of an input image and is used to find the template within the given search image.
    template_img = original_img[portion[0]:portion[1],portion[2]:portion[3]]

    # Apply Template Matching
    # Apply Template Matching Techniques like Normalized Cross Correlation, Cross Correlation, Sum of Squared
    # Difference
    # Try sum of abslolute difference
    #The difference between 2 matrices is computed
    n_rows = original_img.shape[0]
    n_cols = original_img.shape[1]
    n_rows_template = template_img.shape[0]
    n_cols_template = template_img.shape[1]

    # List for collecting metrics of absolute differences
    absolute_differences = []
    coordinate = []
    metrics_abs_diff_map = np.zeros([n_rows-n_rows_template,n_cols-n_cols_template],'float64')

    # Loop by taking in consideration that we cannot go outside the image, if not working use conditional if to check
    for a in range(n_rows-n_rows_template):
        for b in range(n_cols-n_cols_template):

            coordinate.append([a,b])

            # Extract original image sub-image to compute absolute difference, with higher left corner coordinates (a,b)
            sub_img = original_img[a:a+n_rows_template,b:b+n_cols_template]

            # Convert to float64 to perform difference
            sub_img = np.array(sub_img, dtype=np.float64)
            template_img = np.array(template_img, dtype=np.float64)

            # Compute absolute difference between sub-image and template, minimum value will be found template
            # Local minima will be found templates. Store values in a list
            abs_diff = np.sum(np.abs(sub_img - template_img))
            absolute_differences.append(abs_diff)

            # Append in 2D the metric value for later display
            metrics_abs_diff_map[a, b] = abs_diff

    # Match the Images
    # Then match the images with the original image.
    # Find minimum in the map, this time we are in search of the minimum absolute difference!
    ij = np.unravel_index(np.argmin(metrics_abs_diff_map), metrics_abs_diff_map.shape)
    x, y = ij[::-1]

    # Display the result
    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

    # This is the coin, the template
    ax1.imshow(template_img, cmap=plt.cm.gray)
    ax1.set_axis_off()
    ax1.set_title('template')

    # This is the full image
    ax2.imshow(original_img, cmap=plt.cm.gray)
    ax2.set_axis_off()
    ax2.set_title('grayscale image')

    # highlight template coin with rectangle
    hcoin, wcoin = template_img.shape  # Extract the template size
    rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r',
                         facecolor='none')  # Create red rectangle to display around the template coin
    ax2.add_patch(rect)  # Adding the rectangles to full image plotted above

    ax3.imshow(metrics_abs_diff_map)
    ax3.set_axis_off()
    ax3.set_title('`match_template`\nresult')
    # highlight matched region
    ax3.autoscale(False)
    ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    plt.show()


if __name__ == "__main__":

    print('### Task 6 <-----------------------------')

    # Select the Original Image
    # First we select the original image. The image will be in
    # file formats such as  JPG/JPEG, PNG etc.
    #filename_path = os.getcwd()
    # filename = os.path.join(filename_path,'coins.jpg')
    #filename = os.path.join(filename_path, 'coins.jpg')
    #original_image = io.imread(filename)

    # Retrieve from the library standardized image object
    original_image = data.coins()  # image from skimage library

    portion = [170,220,75,130]
    template_matching(original_image, portion)
