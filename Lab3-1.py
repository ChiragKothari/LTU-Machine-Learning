# Lab 3: Classification with K-means clustering and Hyper-parameter optimization using grid search
# Implement a K-means clustering classifier and apply it to a new dataset.
# Additionally, you will do hyper-parameter optimization to find the optimal number of clusters.
import numpy as np
# Imagine you work in a real state company, and you are performing a study to classify the region's rent
# by the average income of their population for an upcoming campaign.
# In this clustering analysis you will not divide the data in validation and train datasets think about the
# reason why you are not doing it in this case.

import pandas as pd
from joblib.parallel import TASK_ERROR
from matplotlib import pyplot as plt
from statistics import mean

def load_csv(path):
    """
    Loads the data from csv file and stores them in pandas dataframe
    Input:
        path : string
            path to the .csv datafile
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
    lines[1:] = [[int(data_row[0]),(data_row[1]),int((data_row[2].split()[0])),int(data_row[3]),float(data_row[4])] for data_row in lines[1:]]
    # Set the columns headers to the values of first row
    new_column_head = [] #empty list
    for ch in lines[0]:
        new_head = input(f"Header: {ch} -- Input new header: ")
        new_column_head.append(new_head)
    # Build dataframe from list of lists, by excluding header row and assigning names of columns
    dfr = pd.DataFrame(lines[1:],columns = new_column_head)
    # Return the pandas dataframe with data ordered
    return dfr

def KMeans(df, num_clust, N, random_state_centroids, interactive_mode_plot = True, interactive_mode_pause = 0.5):
    """
    KMeans algorithm implemented from scratch
    Input:
        df : dataframe
            pandas dataframe containing data
        num_clust : int
            number of clusters estimated to be correct (defining number of centroids)
        N : int
            number of iterations to repeat the clustering process
        interactive_mode_plot : bool
            True to make scatterplot update automatically, False otherwise
        interactive_mode_pause : float
            Pause to set in interactive mode
    Output:
        df_interest : pandas dataframe
            pandas dataframe, with labelled data
        centroids : pandas dataframe
            dataframe with centroids,
    """

    print(f'\n\n###################\nKMeans ALGORITHM\n\nParm:\nNumber of clusters: {num_clust}\nIterations: {N}\n\n###################')

    # Extract subdataframe of interest to perform KMeans
    df_interest = df.iloc[:, 3:] # df containing annual_rent_sqm and average_year_income
    # In the dataframe of interest, initialize a column "Label" of None that will be filled with index of centroid for classification
    df_interest["Label"] = None

    # Extract rent and income data in a list, to later plot data on scatterplot
    rent = df_interest.iloc[:, 0].tolist()
    income = df_interest.iloc[:, 1].tolist()

    # Initialize centroids with starting number of clusters you consider correct, the
    # centroids can be selected as a random point among your sample.
    # Extracting two random rows
    rdm_samples = df_interest.sample(n=num_clust, random_state=random_state_centroids)
    # Convert values into floats for storing average values
    centroids = (rdm_samples.iloc[:, :2]).astype(float)
    centroids["Label"] = [nc for nc in range(num_clust)]

    # Repeat the classification process N (around 10) number of iterations, until mean of the new
    # cluster does not change from the previous iteration.
    convergence = False

    if interactive_mode_plot:
        # Enable interactive mode
        plt.ion()

    for i in range(N):
        print(f'\nITERATION {i+1}  <=====')
        print(f'####\nStarting Centroids')
        print(centroids)

        # Create dictionary where keys are idx of centroids and values are set to empty list
        # At every iteration, list is filled with points associated to centroids and converted to dataframe
        # Reset the dictionary of centroids
        centroid_dict = {key: [] for key in range(len(centroids))}

        # For every point in the dataset
        # Find which point belongs to which cluster, by finding closer centroid to every point (Euclidean distance).
        for p in range(len(df)):
            # Initialize empty list to collect Euclidean distances for point p
            EuclDist_p = []
            # For every centroid
            for c in range(len(centroids)):
                # Compute the Euclidean distance of point p from centroid c
                EuclDist_p.append(((centroids.iloc[c,0] - df_interest.iloc[p,0])**2 + (centroids.iloc[c,1] - df_interest.iloc[p,1])**2)**0.5)

            # Once all Euclidean distances are computed for point p (one for every actual centroid), assign point p
            # to centroid for which Euclidean distances is minimum

            # i) Determine which is the centroid by computing idx of minimum Euclidean distance
            minIdx = EuclDist_p.index(min(EuclDist_p))

            # ii) Store idx of centroid as classification attribute (cluster label) for point p
            df_interest.iloc[p, 2] = minIdx

            # iii) Append point of interest (point p) to list associated with found centroid
            centroid_dict[minIdx].append(df_interest.iloc[p].tolist())

        # Once all points are stored in the lists of the centroids, convert lists to DataFrames
        for idx in centroid_dict:
            centroid_dict[idx] = pd.DataFrame(centroid_dict[idx], columns=["rent","income","label"])

        # Update the centroids
        # At every iteration, new centroid is computed as mean of points in the cluster
        # i) Copy centroids for comparison
        old_centroids = centroids.copy()
        # ii) update centroids
        for c in range(len(centroids)):
            # For every centroid c access the key field to take out the dataframe and compute the mean
            actual_dataframe = centroid_dict[c]
            centroids.iloc[c,0] = mean((actual_dataframe.iloc[:,0])) # rent
            centroids.iloc[c,1] = mean(actual_dataframe.iloc[:,1]) # income
        # Display updated centroids
        print(f'Updated Centroids')
        print(centroids)
        # iii) Comparison of previous centroids with updated centroids
        if old_centroids.equals(centroids):
            # Mean of new cluster does not change from the previous iteration
            convergence = True
            print("\n ===> Centroids as mean of clusters did not change from previous iteration!")
        print('####')

        # Set a different color for each scatter point according to label, visualize what happens at every iteration.
        label = df_interest.iloc[:, 2].tolist()

        plt.scatter(rent,income,c=label)
        # Place centroids
        plt.scatter(centroids.iloc[:,0],centroids.iloc[:,1], s = 100)
        plt.xlabel('Annual rent sqm')
        plt.ylabel('Average yearly income [Ksek]')
        if convergence:
            plt.title(f'KMeans K={num_clust} -- Iteration {i+1} -- Convergence!')
        else:
            plt.title(f'KMeans K={num_clust} -- Iteration {i+1}')

        if interactive_mode_plot:
            plt.draw()  # Update the plot
            plt.pause(interactive_mode_pause)  # Pause for a short interval to allow the plot to refresh
            plt.clf()  # Clear the figure for the next iteration

        else:
            plt.show()

    if interactive_mode_plot:
        plt.ioff()

    return df_interest, centroids

def a(i,df_labelled):
    """
    Calculates the average intra-cluster distance from any point p, defined by index i in the dataframe.
    To calculate the distance, provide the dataframe to extract also the
    Input:
        i : int
            index row of the pandas dataframe corresponding to point of interest p
        df_labelled : pandas dataframe
            pandas dataframe containing all labelled data with Kmeans
    Output:
        intra_cd_avg : float
             average intra-cluster distance from any point i
    """

    ####################################################################### put in iterator
    # extract the label for the i-th point p from the database, plus rent and income values
    label_p = df_labelled.loc[i,"Label"]
    rent_p = df_labelled.iloc[i,0]
    income_p = df_labelled.iloc[i,1]
    # extract all the points with the same label from the pandas dataframe (cluster points)
    # We get only the rows of dataframe where label corresponds to label_p
    df_label = df_labelled[df_labelled["Label"]==label_p]


    # Calculates average intra-cluster distance from any point p
    distances_p = []
    for idx, pt in df_label.iterrows():
        rent_pt = pt.iloc[0]
        income_pt = pt.iloc[1]
        distances_p.append(((rent_p - rent_pt) ** 2 + (income_p- income_pt) ** 2) ** 0.5)

    # Compute average of distances
    intra_cd_avg = mean(distances_p)
    return intra_cd_avg

def b(i,df_labelled,centroids):
    """
    Function that calculates the average inter-cluster distance to the closets cluster
    from p, datapoint i.
    Input:
        i : int
            index row of the pandas dataframe corresponding to point of interest p
        df_labelled : pandas dataframe
            pandas dataframe containing all labelled data with Kmeans
        centroids : pandas dataframe
            pandas dataframe containing labelled centroids obtained with Kmeans algorithm
    Output:
        inter_cd_avg : float
             average inter-cluster distance from any point i
    """

    ####################################################################### put in iterator
    # extract the label for the i-th point p from the database, plus rent and income values
    label_p = df_labelled.loc[i, "Label"]
    rent_p = df_labelled.iloc[i, 0]
    income_p = df_labelled.iloc[i, 1]
    ####################################################################### put in iterator

    # Find the closest cluster to the point i, excluding the centroid to which point already pertains to
    # Adjustement if we have only one cluster around one centroid (for the task of gridsearch where 1 cluster is used)
    if len(centroids) == 1:
        centroids_around_p = centroids
    else:
        centroids_around_p = centroids[centroids["Label"] != label_p]

    centroids_around_p_enlarged = centroids_around_p.copy()
    centroids_around_p_enlarged.loc[:, "Distance"]=None

    for idx, ctd in centroids_around_p_enlarged.iterrows():
        rent_ctd = ctd.iloc[0]
        income_ctd = ctd.iloc[1]
        distance = ((rent_p - rent_ctd) ** 2 + (income_p - income_ctd) ** 2) ** 0.5
        centroids_around_p_enlarged.loc[idx,"Distance"] = distance

    # Find index of centroid (row) with minimum distance (indexes preserved from original DF)
    idxMin = centroids_around_p_enlarged["Distance"].idxmin()
    # Take label of that centroid (for which distance is minimum)
    label_closest_centroid = centroids_around_p_enlarged.loc[idxMin,"Label"]
    # Extract all the points with the same label of the found closest centroid from provided dataframe
    # We get only the rows of dataframe where label corresponds to idxMin
    df_pts_near_centroid = df_labelled[df_labelled["Label"] == label_closest_centroid]

    # Compute all the distances from point p to points pertaining to closest cluster to point p
    distances_p_pts_near_centroid = []
    for idx, pt_near_centroid in df_pts_near_centroid.iterrows():
        rent_pt_near_centroid = pt_near_centroid.iloc[0]
        income_pt_near_centroid = pt_near_centroid.iloc[1]
        distances_p_pts_near_centroid.append(((rent_p - rent_pt_near_centroid) ** 2 + (income_p - income_pt_near_centroid) ** 2) ** 0.5)

    # Return the average inter-cluster distance
    inter_cd_avg = mean(distances_p_pts_near_centroid)
    return inter_cd_avg

def Silhouettes_Database(df_labelled,centroids,verbose = False):
    """
    Function that iterates over all points i in the dataset and calculates S(i)
    Input:
        df_labelled : pandas dataframe
            pandas dataframe containing all labelled data with Kmeans
        centroids : pandas dataframe
            pandas dataframe containing labelled centroids obtained with Kmeans algorithm
        verbose : Boolean
            If True, plot every silhouettes score for every datapoint in the dataset
    Output:
        S : float
             silhouette coefficient, as given by the average of all the silhouette coefficients for all the data point i
    """

    # Compute Silhouettes scores for every point
    if verbose:
        print('\n\nSilhouettes scores S(i) for all points of labelled database:\n')
    silhouette_scores_points = []
    # Loop through the database to compute the silhouettes coefficients for every point, S(i) and append it
    for i, pts in df_labelled.iterrows():
        a_i = a(i, df_labelled)
        b_i = b(i, df_labelled, centroids)
        #silhouette score for data point i
        S_i = (b_i - a_i)/max(a_i,b_i)
        if verbose:
            print(f'Point {i}, S({i}) = {S_i}')
        silhouette_scores_points.append(S_i)

    S = mean(silhouette_scores_points)
    if verbose:
        print(f'\nSilhouettes average score S: {S}\n')

    return S

def classify_new_point_plot(centroids,new_points,df_labelled):
    """
    Function that accepts centroids from trained KMeans model, a list of new points to
    evaluate and returns the predicted labels. It also plots and the labelled dataframe used to train the model
    Input:
        centroids : pandas dataframe
            Contains labelled centroids obtained with Kmeans algorithm
        new_points : list
            Contains points to evaluate
        df_labelled : pandas dataframe
            Contains all labelled data with Kmeans
    Output:
        S : float
             silhouette coefficient, as given by the average of all the silhouette coefficients
    """

    # Lists for plotting
    rents_new_p_plotting = []
    incomes_new_p_plotting = []
    # Distances list
    distances_p = []
    # Centroid label list
    centroid_label = []
    # List for predicted values
    y = []

    for p in new_points:
        # Points attribute, extract for distance computation and append for new plotting
        rent_p = p[0]
        rents_new_p_plotting.append(rent_p)
        income_p = p[1]
        incomes_new_p_plotting.append(income_p)
        for idx, ct in centroids.iterrows():
            # Attributes of centroid
            rent_ct = ct.iloc[0]
            income_ct = ct.iloc[1]
            distances_p.append(((rent_p - rent_ct) ** 2 + (income_p - income_ct) ** 2) ** 0.5)
            centroid_label.append(ct.iloc[2])
        print(f"####\nPoint: {p}")
        print(f"Distances computed: {distances_p}")
        # Extract index of minimum distance to take the class of cluster
        idx_centroid_class = distances_p.index(min(distances_p))
        class_label_p = centroid_label[idx_centroid_class]
        # Append to list the class predicted for the point
        y.append(class_label_p)
        print(f"Prediction: {class_label_p}")

        # Empty lists for next iteration with new point to re-start appending
        distances_p = []
        centroid_label = []

    # Print Prediction
    print(f"Input points: {new_points}\nPrediction array: {y}")
    # Visualization
    # Set a different color for each scatter point according to label, visualize what happens at every iteration.
    rent = df_labelled.iloc[:,0]
    income = df_labelled.iloc[:,1]
    label = df_labelled.iloc[:, 2].tolist()

    # Scatter dataset
    plt.scatter(rent, income, c=label)
    # Place centroids
    plt.scatter(centroids.iloc[:, 0], centroids.iloc[:, 1], s=100)


    # Scatter new points in clusters.
    # c = y is messing up with labels
    plt.scatter(rents_new_p_plotting, incomes_new_p_plotting, c='b', edgecolor='red', s=50)

    plt.xlabel('Annual rent sqm')
    plt.ylabel('Average yearly income [Ksek]')

    plt.title(f'KMeans K={len(centroids)}')
    plt.show()

    return y

if __name__ == "__main__":

        # Load the dataset Average rent in a rental apartment by year and region
        # ---> 1.1 Load the new dataset found in the canvas page for this Lab: rent_vs_inc.csv the same way you did in the last lab
        df = load_csv('inc_vs_rent.csv')
        print(df.head())
        # What information can you get by just looking at the table? That year is 2020 and every region is unique and representable by a number

        # ---> 1.2 Start by creating a scatter plot for your points.
        annual_rent_sqm = df.iloc[:,3]
        average_year_income = df.iloc[:,4]
        plt.plot(annual_rent_sqm, average_year_income, "b.")
        plt.title('Average yearly income VS annual rent sqm')
        plt.xlabel('Annual rent sqm')
        plt.ylabel('Average yearly income [Ksek]')
        plt.show()

        # ---> 1.3 create K-means function from scratch, plot the classified points interactively
        num_clust = 2
        N = 10
        random_state_centroids = 1
        [df_labelled,centroids] = KMeans(df, num_clust, N, random_state_centroids, interactive_mode_plot=True, interactive_mode_pause=0.5)
        print('Labelled Dataframe\n',df_labelled)

        # ---> Task 2: Hyper-parameter optimization
        # Implement grid search to find the optimal number of k-means clusters for this classification task.

        # ---> 2.1  In order to perform hyper-parameter optimization for number of clusters, use
        # the silhouette score. Create a function that finds the silhouette coefficient for a
        # grid of values between 1 and 10.
        # Silhouette score [-1; 1]
        # 1 = observation is well clustered, 0 = observation on the boundary, <0 = observation in wrong cluster

        # Create a function a, b and main function Silhouettes_Database, where functions a and b are used iteratively
        # to compute the silhouettes score for every point
        Silhouettes_Database(df_labelled, centroids)

        # Perform grid search obtaining average S(i) for all points of each cluster value in grid
        # to get the cluster’s silhouette coefficient. (Iterate over the cluster values between 1 to 10)

        grid_num_clusters = list(range(1,11))

        # List to collect Silhouettes for grid evaluation
        grid_S = []

        for num_clusters in grid_num_clusters:
            # Train KMeans
            [df_labelled, centroids] = KMeans(df, num_clusters, N, random_state_centroids, interactive_mode_plot=True,
                                              interactive_mode_pause=0.5)
            S = Silhouettes_Database(df_labelled, centroids)
            print(f"\nNumber of clusters: {num_clusters} -- Silhouettes score: {S}")

            grid_S.append(S)

        # Graph the cluster’s silhouette coefficient for each value in the grid to determine the
        # optimal number of clusters for this exercise.

        max_S_idx = grid_S.index(max(grid_S))
        bestK = grid_num_clusters[max_S_idx]

        plt.plot(grid_num_clusters,grid_S,'-o')
        plt.axvline(x=bestK, color='red', linestyle='-')
        plt.grid(True)
        plt.title(f"Silhouettes coefficients grid search -- Best K= {bestK}")
        plt.xlabel('Number of clusters K')
        plt.ylabel('Silhouettes coeff. score')
        plt.show()

        # ---> 2.2 Create a scatter plot with the cluster colors for the newly found optimal number of clusters.
        # Again, plot best K
        [df_labelled, centroids] = KMeans(df, bestK, N, random_state_centroids, interactive_mode_plot=True,
               interactive_mode_pause=0.1)

        # ---> 2.3 Finally evaluate how your created model predicts new values:
        # For this task we will assume we have 3 unnamed regions with the following annual rent
        # and average salary [1010, 320.12], [1258, 320], [980, 292.4]

        # Find which cluster these data points belong to and plot them in the graph.
        # Let's try model for the best K
        new_points = [[1010, 320.12], [1258, 320], [980, 292.4]]
        classify_new_point_plot(centroids,new_points,df_labelled)

        # Based on the cluster graph do you think your model successfully predicted the cluster
        # for these values?

