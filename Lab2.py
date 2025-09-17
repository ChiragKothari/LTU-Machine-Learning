# For this lab you will load a new dataset and a subset of that dataset. The data to be used is available in
# Canvas and has been downloaded from the SCB website (www.scb.se). The full dataset includes mean
# income by region, age and year while the provided subset contains the average income by age from 20
# to 50 years only.
# The objective of this lab is to conduct regression analysis. The focus will be on performing multiple linear
# regressions using the same dataset to assess their effectiveness in various segments. Additionally, you
# will explore alternative regression techniques to address the dataset, followed by a reflective analysis of
# these options.


# ----> Task 1: Load first a subset of the data to be analyzed
# Create a function to load data from a csv file based on basic file I/O functions described in the lectures.
# (Do not use pandas or a csv library to load the data, though you can use pandas to save the result).
# Using that function, load the file inc_subset.csv that is available in Canvas. Inspect the values of the
# loaded data elements using the debugger and print them on screen (you should use both the debugger
# and the printout to verify that the csv loader behaves as expected).
# This dataset contains information regarding the average income by age from 20 to 50 years.

import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def load_csv(path,dataset_part):
    """
    Loads the data from csv file and stores them in pandas dataframe
    Input:
        path : string
            path to the .csv datafile
        dataset_part : boolean
            If True process dataset part, If false process the whole dataset
    Output:
        pandas dataframe collecting input data
    """
    # Open the file in read mode to retrieve data
    f = open(path,'r')
    # Store line by line the data from the dataset in a list
    lines = f.readlines()
    # We close the file
    f.close() # set debug point at this line to inspect the "lines" datatype and its content, in the "Threads & Variables"
    # Inspect first five strings (lines) in the list from imported the dataset, to visually check they conform to
    # content of .csv file by printing
    for i in range(5):
        print(f'Line # {i}: {lines[i]}')
    # We can see that data conform to .csv file content.
    # First line is column header
    # Second line is first instance of dataset, third line is second instance of dataset..

    # Build pandas dataframe out of the input data, by manipulating and processing
    # Every string has data separated with a ','
    # Create a list containing instances represented as a list of strings, with data splitted
    lines = [data_row.split(',') for data_row in lines]

    if dataset_part:
        # Convert from string to int the first and second element of each sublist, convert to float after
        # stripping out the \n from third element of each sublist
        lines[1:] = [[int(data_row[0]),int(data_row[1]),float(data_row[2].strip('\n'))] for data_row in lines[1:]]
    else:
        # Process whole dataset
        lines[1:] = [[int(data_row[0]),(data_row[1]),int((data_row[2].split()[0]).replace('+','')),float(data_row[3])] for data_row in lines[1:]]
    # Set the columns headers to the values of first row
    new_column_head = [] #empty list
    for ch in lines[0]:
        new_head = input(f"Header: {ch} -- Input new header: ")
        new_column_head.append(new_head)
    # Build dataframe from list of lists, by excluding header row and assigning names of columns
    df = pd.DataFrame(lines[1:],columns = new_column_head)

    # Return the pandas dataframe with data ordered
    return df

def linear_regression_model(X,y):
    """
    Applies linear regression Model as from vectorized functions presented in Lecture 8 and chapter 4 of  ML course book
    Input
        X : 1Dim np array
            instance’s feature vector
        y : 1D np array
            vector of target values
    Output
        theta_parm_optimized : 1D np array
            model's parameter vector, containing the coefficients theta0, theta1, theta2, ...
    """
    X_ = np.c_[np.ones((len(X),1)), X]  # add x0 = 1 to each instance. np.c_ is used to concatenate the 2 arrays if they were 2 columns
    theta_parm_optimized = np.linalg.inv(X_.T.dot(X_)).dot(X_.T).dot(y)
    print(f'Linear regression model: y = b + ax, with b = {theta_parm_optimized[0]} a = {theta_parm_optimized[1]}')

    return theta_parm_optimized

def MSE(X,coeff,y,full=False,yp=None):
    """
    Computes the MSE metric for evaluating each of the real values against the predicted values
    Input:
        X:numpy array
            real values used to perform prediction
        coeff:numpy array
            coefficient for the linear regression
        y:numpy array
            real value array
        full:boolean
            default is set to False,so it will skip computing prediction yp and use values from validation
        yp:numpy array
            predicted values that might be passed in case available
    Output:
        MSE
    """
    if full:
        # Prepare data for validation of model
        x_to_validate = np.c_[np.ones((len(X), 1)), X]  # add x0 = 1 to each instance
        yp = x_to_validate.dot(coeff)

    # Compare predicted values with real values with MSE: sum of square of difference between predicted yp and real y averaged by the number of instances
    mse = (sum((yp-y)**2))/len(y)

    print(f'MSE for the linear regression: {mse}')

    # Returning MSE
    return mse


def split_convert_dataset(df, valid_prop = 0.2, rand_seed = 42, dataset_part = True):
    """
    Split the provided dataframe into validation set and training set.
    Returns 4 numpy arrays (2 of ages and 2 of incomes) to be used for the regression model
    Input:
        df:pandas dataframe
            dataframe containing ages and incomes
        valid_prop: float
            proportion of data devoted to validation, 1-valid_prop is left to training
        rand_seed: int
            random state for random sampling (for reproducibility)
        dataset_part: bool
            True if it is only part of full dataset (first part of lab), False when used for full dataset
    Output:
        ages_train: np array
            1D array containing ages data for training
        incomes_train: np array
            1D array containing income data for training
        ages_valid: np array
            1D array containing ages data for validation
        incomes_valid: np array
            1D array containing income data for valiadation

    """
    valid_data = df.sample(frac=valid_prop, random_state=rand_seed)  # repetitive with random seed
    train_data = df.drop(valid_data.index)
    # Get numpy arrays for training the model from pandas dataframe
    if dataset_part:
        i = 1
        j = 2
    else:
        i = 0
        j = 1

    ages_train = np.array((list(train_data.iloc[:, i])))
    incomes_train = np.array((list(train_data.iloc[:, j])))

    # Get numpy arrays for later validation
    ages_valid = np.array((list(valid_data.iloc[:, i])))
    incomes_valid = np.array((list(valid_data.iloc[:, j])))

    return ages_train, incomes_train, ages_valid, incomes_valid

def scatterplot(ages,incomes,cf_training):
    """
    Generates scatterplot and computes and plots regression line on top of it
    Input:
        ages:np array
            1D array containing ages
        incomes:np array
            1D array containing incomes
        cf_training:np array
            1D array containing coefficients to plot regression line
    """
    # Create densely populated feature array to evaluate the lineat model and get predicted incomes for visualization
    ageLine_list_X = np.linspace(min(ages), max(ages), int(max(ages) - min(ages) + 1))
    ageLine_X = np.array(ageLine_list_X)
    xfit_c = np.c_[np.ones((len(ageLine_X), 1)), ageLine_X]  # add x0 = 1 to each instance
    incomeLine_y = xfit_c.dot(cf_training)
    # Plot the data (scatterplot + regression line)
    plt.plot(ageLine_X, incomeLine_y, "r-")
    plt.plot(ages, incomes, "b.")
    plt.title('Income from Age in Sweden with regression line')
    plt.xlabel('Age [year]')
    plt.ylabel('Average Income')
    plt.show()

def validate_regression(ages_valid,incomes_valid,cf_train):
    """
    Validates the regression model
    Input:
        ages_valid:np array
            1D array containing ages for validation evaluation
        incomes_valid:np array
            1D array containing incomes for validation
        cf_train:np array
            1D array containing coefficients of the regression model
    """
    xfit_c_valid = np.c_[np.ones((len(ages_valid), 1)), ages_valid]  # add x0 = 1 to each instance
    incomes_predicted_valid = xfit_c_valid.dot(cf_train)
    print(f'Incomes real values -- validation\n{incomes_valid}')
    print(f'Incomes predicted values -- validation\n{incomes_predicted_valid}')
    print(f'Absolute error prediction -- validation\n{abs(incomes_valid - incomes_predicted_valid)}')
    print(f'Percentage error prediction-- validation\n{(abs(incomes_valid - incomes_predicted_valid) * 100) / incomes_valid}')




if __name__ == "__main__":

    execute_task_3 = True

    if not execute_task_3:

        #path_full_data = 'inc_utf.csv'
        path_full_data = 'inc_subset.csv'
        # ----> Task 1
        print('\n##Task 1')
        dataframe = load_csv(path_full_data,True)

        # Discussion: Discuss with your lab partner. When is it more appropriate to use the debugger for
        # inspecting variable values, and when would you prefer to use a print function? Identify:
        # • One case when it is more convenient/efficient to use the debugger: Pycharm debugger has
        # "View as Dataframe" in debugger mode that is very powerful to visualize the data in dataframe. Debugging is also
        # very helpful to inspect the datatype of the imported data and with more complex situations
        # • One case when it is more convenient/efficient to use a printout: The print function is faster and can give insight on smaller data
        # e.g. quickly printout first line of imported dataset to check the header and the value formats.

        print('##Task 2')
        #  ----> Task 2: Linear regression on a data subset.
        # For this task you will perform a linear regression on the average yearly income and age data of the year
        # 2020. Develop the model from scratch using the vectorized functions presented in Lecture 8 and chapter
        # 4 of the ML course book (do not use sklearn).

        # see linear_regression_model

        # IDs = np.array((list(dataframe.iloc[:, 0])))
        Ages = np.array((list(dataframe.iloc[:, 1])))
        Incomes = np.array((list(dataframe.iloc[:, 2])))
        print('\nLINEAR REGRESSION MODEL WITH WHOLE DATASET') # still on subset of data from the website
        coeff = linear_regression_model(Ages, Incomes)

        # -----> Task 2.1 Perform the linear regression on the provided dataset
        # We want to split into training and validation to assess if the linear regression is generalizing properly
        # Discuss whether you should take a random sample in this case
        # We might use random sampling, since we do not know if data are ordered in a particular way in the dataset and
        # if we take just the first data as training and last as validation, we might get a bias.

        Ages_training, Incomes_training, Ages_validation, Incomes_validation =  split_convert_dataset(dataframe, valid_prop=0.2, rand_seed=42, dataset_part=True)

        # Compute coefficients (training data)
        print('\n2.1 LINEAR REGRESSION MODEL FROM ONLY TRAINING DATASET')
        coeff_training = linear_regression_model(Ages_training, Incomes_training)

        # ----> Task 2.2 Create a scatter plot for your data and plot the regression line.
        print('\n2.2 Scatterplot is displayed for the model trained with training dataset')
        scatterplot(Ages, Incomes, coeff_training)

        # ----> 2.3  Predict the values for the validation dataset and print both the real values and the predicted ones. How do they look?
        print('\n2.3 LINEAR REGRESSION MODEL VALIDATION:')
        validate_regression(Ages_validation, Incomes_validation, coeff_training)

        # ----> 2.4  Evaluate the model using MSE, implement the function that evaluates each of the real values
        # against the predicted values (do not use sklearn MSE functions).
        # MSE functions is developed above
        print('\n2.4 MSE Evaluation of model:')
        mse = MSE(Ages_validation, coeff_training, Incomes_validation, full=True)
        print(f'RMSE Evaluation of model:{mse**0.5}')

        # ----> 2.5 Explain the obtained MSE value and what does it mean.
        # The MSE is the average of all the squared errors for all the datapoints (real value versus model-predicted value)
        # Given that errors are squared, larger errors weight more, lower errors weight less.
        # To make the MSE comparable to the data, square root (RMSE) might be computed, to compare it to the same data unit
        # Do you think linear regression is a good approach for this dataset?
        # We think that liner regression for this data might be used even though MSE and RMSE are not very low.
    else:
        # ----> Task 3
        # With the load function created earlier, load the second table inc_utf.csv that is available in Canvas. This
        # dataset contains information regarding the average income by age and region.
        # Check that the data has been loaded successfully using the debugger and a printout on screen.

        path_full_data = 'inc_utf.csv'
        dataframe = load_csv(path_full_data,dataset_part=False)
        print(f'There are {len(dataframe)} instances in the database')

        print('\nTask 3.1')
        # -----> 3.1: To execute the linear regression as you did in the previous task, you'll need to group the data by
        # age groups to find the mean across all regions. A recommended approach is to utilize the pandas
        # group_by function. Inspect the data to see that now you have way more age groups that
        # in the previous task. Additionally, some fields need to be cleaned e.g., the “years” column need
        # to be converted to an integer and the text that is not number has to be removed.
        #Let's name headers as age and region
        groups_age_mean_income_regions = dataframe.groupby(by='age')['Income'].mean() # Group by ages by averaging the income over the regions
        print(groups_age_mean_income_regions) # (85 age groups!)

        print('\nTask 3.2')
        # 3.2: Perform the linear regression on the provided dataset. Use validation and train data split.
        # Convert to dataframe to perform the split and extracting np arrays to perform the regression

        # Convert the series into dataframe and rename the columns' headers
        dataframe_grouped_age_mean_income_region = groups_age_mean_income_regions.reset_index()
        dataframe_grouped_age_mean_income_region.columns = ['Age', 'Mean Income']
        # Split data into training and validation with random seed, returns directly np array to evaluate in model
        Ages_training, Incomes_training, Ages_validation, Incomes_validation = split_convert_dataset(dataframe_grouped_age_mean_income_region, valid_prop=0.2, rand_seed=42, dataset_part=False)
        # Perform the linear regression on the provided dataset
        cf_full_train = linear_regression_model(Ages_training, Incomes_training)

        # 3.3: Create a scatter plot for your data and plot the regression line, from np arrays
        Ages = np.array((list(dataframe_grouped_age_mean_income_region.iloc[:, 0])))
        Incomes = np.array((list(dataframe_grouped_age_mean_income_region.iloc[:, 1])))
        scatterplot(Ages, Incomes, cf_full_train) # from the scatterplot we can see that the data are NOT linear

        print('\nTask 3.4')
        # 3.4: Predict the values for the test dataset and print both the real values and the predicted ones.
        # How do they look? They look much higher in percentage to those from previous lab
        validate_regression(Ages_validation, Incomes_validation, cf_full_train)

        print('\nTask 3.5')
        # 3.5: Evaluate the model using MSE as you did with the previous exercise.
        mse = MSE(Ages_validation, cf_full_train, Incomes_validation, full=True)
        # 3.6: Explain the obtained MSE value and compare it to the previous task.
        # MSE for the linear regression: 11924.661312020917 (too high)

        # ------> Task 4: Reflections.
        # 4.1 What differences can you see from the graphs, predicted values and MSE scores from both linear regressions?
        # In task with small dataset the line approximates the data quite well (MSE lower), with full dataset the line does not represent the distribution at all (MSE huge)
        # 4.2 How do you think this analysis can be improved considering the obtained results of task 3.
        # Polynomial regression, changing the model.


        # ------> Task 5: Polynomial regression with hyperparameter tuning
        # In this lab you performed linear regression to find the relationship between age and income. However,
        # the linear regression result may not be the optimal result as shown by the MSE value.
        # You will now use Scikit-learn to perform polynomial regression and evaluate if it is a better model.


        # 5.1 Polynomial regression is a special case of linear regression. Perform polynomial regression. With
        # the use of PolynomialFeatures from sklearn.preprocessing you will be able to increase the
        # number of features the linear regression model is trained.
        # For additional information on how to perform this task check lecture 8 and page 128 of the ML
        # book.

        # Scikit-Learn’s PolynomialFeatures class transforms training data, adding the square (2nd-degree
        #  polynomial) of each feature in the training set as new features.

        # A) increase the number of features......
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        # Transform 'Ages' to array-like of shape (n_samples, n_features)
        # Wants each row to be a sample and each column to be a feature, so we need a 1D column array of features
        # 'Ages' must be reshaped to be a 1 column array of multiple rows (each age in 'Ages' in its own row)
        Ages_reshaped = Ages.reshape(-1, 1) # it is not symbolical formula, but a feature array with instances
        # Expansion with only one feature. Ages_poly contains original feature of Ages plus the square of this feature.
        # Now you can fit a LinearRegression model to this extended training data

        # Split dataset in training and validation do it now to benefit next task as well
        Ages_train, Ages_val, Incomes_train, Incomes_val = train_test_split(Ages_reshaped, Incomes, test_size=0.2,
                                                                                      random_state=42)

        print('\nTask 5.1')
        # Return a transformed version of Ages_train into polynomial features
        Ages_poly_train = poly_features.fit_transform(Ages_train) # creating an expanded dataset with the desired features by following polynomial degree
        print(f'1 linear feature Ages {Ages_train[0]}') # Print first element of original dataset
        print(f'2 linear features [Ages Ages^2] {Ages_poly_train[0]}') # Print first element of expanded dataset

        # B) .....the linear model is trained
        # Fit the linear model with the 2 features dataset now.
        lin_reg = LinearRegression()
        lin_reg.fit(Ages_poly_train, Incomes_train) # We fit the linear regression model with training data
        print(f'Linear regression model Income = c + b*Age + a*Age*Age; c ={lin_reg.intercept_}, [b a] = {lin_reg.coef_}')

        # ----> 5.2 You may notice there is a Hyperparameter for this model, which is the polynomial degree, select
        # at least 4 different values for this parameter and perform the polynomial regression for each.
        # Use MSE to evaluate the model using the validation dataset. For example, you can choose a
        # polynomial degree of 2, 3, 5 and 8.
        print('\nTASK 5.2')
        Poly_regression_dict = {}

        # Fit a linear regression model on the single feature
        lin_reg_1feat = LinearRegression()
        lin_reg_1feat.fit(Ages_train, Incomes_train)
        # Prepare dense array to plot regressors
        ageLine_list_X = np.linspace(min(Ages), max(Ages), int(max(Ages) - min(Ages) + 1))
        ageLine_list_X = ageLine_list_X.reshape(-1,1)
        Incomes_1feat = lin_reg_1feat.predict(ageLine_list_X)

        # Prepare another linear regression model to provide expanded feature (polynomial regression)
        lin_reg = LinearRegression()
        for poly_degree in [2, 3, 5, 8, 10, 12, 16]:
            # Prepare polynomial features for linear regression model with different polynomial degree
            poly_features_degree = PolynomialFeatures(degree=poly_degree, include_bias=False)
            # Expand training data by adding features for polynomial regression
            Ages_poly_train = poly_features_degree.fit_transform(Ages_train)
            # Fit the linear regression model to training data
            lin_reg.fit(Ages_poly_train, Incomes_train)
            # Save parameters of fitted model and MSE computed over the validation dataset
            # Predict the incomes with the inbuilt predict method after expanded validation with transformed model
            Ages_poly_val = poly_features_degree.fit_transform(Ages_val)
            Incomes_predict_val = lin_reg.predict(Ages_poly_val)
            # Compute mean squared error from the same income validation known data
            mse = (sum((Incomes_predict_val - Incomes_val) ** 2)) / len(Incomes_val)
            # Save in dictionary the result for every hyperparameter
            Poly_regression_dict[str(poly_degree)] = [mse, lin_reg.intercept_, lin_reg.coef_]
            print(f'Polynomial model of degree = {poly_degree} --- MSE = {int(mse)}')
            # We can see that MSE on validation dataset decreases with increasing degree, but if degree is too high it
            # can easily overfit and MSE would become high

            # ----> 5.4
            # Graph the results of the polynomial regression line with the optimal degree found along with
            # the linear regression line.

            # Evaluate the dense transformed features array ageLine_list_X_poly into Incomes_Predict_Y through the
            # actual trained lin_reg model
            ageLine_list_X_poly = poly_features_degree.fit_transform(ageLine_list_X)
            Incomes_predict_y = lin_reg.predict(ageLine_list_X_poly)

            # Plot the data (linear regression line + polynomial regression line + data)
            # Plot linear regression (Incomes_1feat evaluated outside the loop)
            plt.plot(ageLine_list_X, Incomes_1feat, "r-")
            # Plot predictor (single feature) against the predicted target value through the polynomial model
            plt.plot(ageLine_list_X, Incomes_predict_y, "g-")
            # We plot the scatter plot of all available values in the dataset
            plt.plot(Ages, Incomes, "b.")
            plt.title(f'Income from Age in Sweden, MSE polynomial = {mse:.1f}')
            plt.xlabel('Age [y]')
            plt.ylabel('Average Income')
            plt.legend(['Linear Regression',f'Polynomial Regression degree: {poly_degree}'])
            plt.show()

        # ----> 5.3 a
        # In this data set you do not use a test set. However, in case you did, which dataset would you use
        # to perform the hyperparameter tuning?
        # Hyperparameter tuning might be performed on the validation set. Test set is used for final evaluation of trained
        # and tuned model, while performing hyperparameter tuning on training dataset might lead to overfitting by adhering too
        # much to training data


        # ----> 5.3 b
        # From your choices, which order of polynomial is best? Discuss and explain.
        # The order of polynomial which is the best might be the one with lowest MSE (8), however by looking at the data,
        # it seems that a less complex model of degree 3 would still provide low MSE and good generalization
        # Using a test set would be nice to evaluate the model. From the plot, degree 8 seems to perform very well.














