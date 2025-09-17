import pandas as pd

# Import developed functions in previous task
from Lab1_Task_1_2 import *

# -----> 3 pandas dataframe, load dataset <---- While debugging in pycharm, we get a lot of interesting insights in the data and the df under 'Threads & Variables' --> "View as DataFrame" to the right
# This dataset provides a set of housing districts and their information.
df = pd.read_csv("housing.csv")

# Extract headers
headers = df.columns
# Convert to list
headers_list = headers.tolist()
# Print number of attributes and visualize them
nCols = df.shape[1]
print('\n\n')
print(f'The number of attributes is: {nCols}')
print(headers_list)
print('\n')

# -----> 4.1 Count the number of districts loaded in this exercise
nRows = df.shape[0]
print(f'4.1 The number of districts/istances in the dataset is: {nRows}')
print('\n')

# -----> 4.2 Calculate the mean of house values among all the districts
# Extract 'median_house_value' column as a list
median_house_value = df['median_house_value'].tolist()

# Compute the mean of the median values as estimate
print(f'4.2 the mean of house values among all the districts is: {Mean(median_house_value)}')
print(f'The minimum of house values among all the districts is: {Min(median_house_value)}') # Can help for point 4.5
print(f'The maximum of house values among all the districts is: {Max(median_house_value)}') # Can help for point 4.5
print('\n')

# -----> 4.3 Create a histogram for amount_of_households, median_income , housing_median_age and, median_house_value.
from matplotlib import pyplot as plt

# Retrieve the columns from the dataframe as lists to simplify processing
households = df['households'].tolist() #6
median_income = df['median_income'].tolist() #7
housing_median_age = df['housing_median_age'].tolist() #2

# create two paired lists for easiness
attributes_interest_data = [households, median_income, housing_median_age, median_house_value]
title_interest_dataframe = ['households', 'median_income', 'housing_median_age', 'median_house_value']

# Plot consequently, to discuss every single histogram with right title
for idx,attribute in enumerate(attributes_interest_data):
    # Crates plot with histogram, title and display it
    plt.hist(attribute)
    plt.title(f'histogram {title_interest_dataframe[idx]}')
    plt.show()


# -----> 4.4 What do you notice about the graphs? Specifically focus on end of 'housing_median_age' and 'median_house_value' graphs
# Consideration: histogram helps us visualize distribution of quantitative data
# end of 'housing_median_age' --> the end of the histogram tells us that around 2000 districts (frequency, y axis) out of 20640 have houses where the median age is 50yrs (x-axis). Median is stronger than mean to outliers
# 'median_house_value' --> The median house value is displaying us most houses are worth between 100.000 and 250.000
#                          The end of the histogram is showing us that there are around 1250 houses worth 500.000

# -----> 4.5 What do you think about the magnitude of the values in median_house_value? What may have
# happened to them in the processing, think about the units.

# a) Prior to calling the matplotlib pyplot method 'hist'
# The data is provided as a list of floats: [452600.0, 358500.0, ...] <--- median_house_value,
# median_house_value  retrieved from imported dataframe df['median_house_value'].tolist().
# The dataframe as imported, reported values as %s. Can be seen while debugging in Pycharm and visualizing the values

# => b) This is suggesting us that processing happened when calling the method 'hist' and providing string of float as input
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
# The method was suggested to be called with only one parameter (the list of float values)

# Below considerations from documentation

# 'bins': The second parameter is the number of bins of the histogram, set by default to 10.
# "If bins is an integer, it defines the number of equal-width bins in the range"
# Therefore, in the visualization, the values from the original list are clustered in 10 equally distributed bins,
# we lose in resolution.

# 'range': The lower and upper range of the bins.
# Lower and upper outliers are ignored.
# If not provided, range is (x.min(), x.max()), with x input list of values

print('4.5')
[n, bins, patches] = plt.hist(median_house_value)
print(f'Values of histogram bins:\n{n}') # <-- The values of the histogram bins
print(f'Sum of values clustered in histogram bins: {sum(n)}') # <--- corresponds to total number of istances in dataset, as expected
print(f'Edges of histogram bins:\n{bins}') # <-- The edges of the bins. Length nbins + 1
print('\n')

# Additional Task: For each ocean proximity category in the dataset calculate the mean house value.
# Hint: To perform this task you will need to group and calculate the mean of all districts per
# “ocean_proximity” category, an example in how to execute this task can be found in the
# picture

# There is a powerful builtin pandas dataframe method that helps us to answer the question
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
# it makes sense to learn the method and exploit it to get insight

# groups_ocean_proximity = df.groupby(by = 'ocean_proximity').mean() <--- Powerful by displaying the different
# ocean_proximity categories and the mean of all the other aggregated attributes
# Create a groupBy object where df is partitioned into groups based on the categories of 'ocean_proximity'.

# If we are interested only in the mean of a particular attribute, in this case the house value, then we consider only 'median_house_value'
# We select attribute on which to perform aggregation 'median_house_value', we compute the mean of the grouped values
groups_ocean_proximity_house_values = df.groupby(by = 'ocean_proximity')['median_house_value'].mean()

print('4.5 Additional task')
# print aggregation <-- it seems the most valuable houses are on the island, while inland houses are the cheapest
print(groups_ocean_proximity_house_values)
print('\n')

# ----> 4.6 Think about the following two cases:

# 1. Let's think about the first task of this lab. Imagine you are a teacher, and you want to analyze
# the performance of your students on a recent exam. You have the exam scores of all the
# students in your class. Which metric is more adequate for this analysis?

# If we need to select a single metric, then MEAN would be the most appropriate, to get feeling of
# how student performed on average.

# Paired with STANDARD DEVIATION, it would help understand how single grades of
# the student are differing from the MEAN, how grades are spread around it.
# Together with  MEDIAN, we might get insights if whether there are outliers or not

# 2. Consider a study on the income distribution of a country. The dataset includes the incomes
# of individuals, and it is known to have a few extremely high-income earners, such as
# billionaires. Which metric would be better in this case?

# If we know about the outlier of extremely high-income earners, then the MEDIAN would be NOT sensitive to that, and it would
# help us to understand the central tendency.
# MEDIAN ABSOLUTE DEVIATION measures the spread of data around MEDIAN and might be used to understand the distribution of
# incomes around the median value.

