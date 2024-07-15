
from migrant_detection import linear_kernel_distance
from migrant_detection import distance_to_plane
from data_generation import generate_gaussian_dataset
from scipy.spatial import distance
import numpy as np
import pandas as pd
import seaborn as sns 
from joblib import dump, load



def overlap_check(data):
    # Check for overlaps within the dataset
    cluster_0 = data[data['cluster'] == 0].drop(columns=['cluster'])
    cluster_1 = data[data['cluster'] == 1].drop(columns=['cluster'])
    intervals = pd.DataFrame(cluster_0.min(), columns=['cluster_0_min'])
    intervals['cluster_0_max'] = cluster_0.max()
    intervals['cluster_1_min'] = cluster_1.min()
    intervals['cluster_1_max'] = cluster_1.max()
    print(intervals.values)
    #[print(x) for x in intervals.values]
    #intervals.head()
    overlaps = [pd.Interval(x[0],x[1]).overlaps(pd.Interval(x[2],x[3])) for x in intervals.values]
    print(overlaps)
    return overlaps


def cluster_vector_distance_variance_threshold(cluster, center1, center2, multiplier):
    """
    Calculate distance thresholds for a cluster based on the variance of the distance to the cluster connecting vector.
    Three Sigma rule states, that around 99,73% of the Gaussian distribution is captured within 3 standard deviations from the mean.

    Args:
    - cluster (Pandas Dataframe): Datapoints in the cluster
    - center1 (list): first cluster representative for cluster connecting vector
    - center2 (list): second cluster representative for cluster connecting vector
    - multiplier: multiplier for standard deviation

    Returns:
    - float: threshold based on the three sigma rule
    """
    distances = [distance_to_plane(center1, center2, x) for x in cluster.values]
    # To visualize the distribution save the distances
    #dump(distances, 'Plots/distances_ccv_' + str(cluster.values[0]) +'.pkl')
    m = np.mean(distances)
    x_std = (np.std(distances) * multiplier)
    return m + x_std

def generate_migrants(data,n):
    """
    Generate a set of migrants 

    Args:
    - data (Pandas dataframe): Dataset used to train our model
    - n (int): number of migrants to generate

    Returns:
    - list: n migrants with 20 steps between start and end and 6 dimensions per step
    """
    # Extract the datapoints which lie in each cluster and drop the predicted label
    cluster_0 = data[data['cluster'] == 0].drop(columns=['cluster'])
    cluster_1 = data[data['cluster'] == 1].drop(columns=['cluster'])
    # Chose random start and endpoints from each cluster 
    group_start = cluster_0.iloc[np.random.choice(cluster_0.shape[0], n, replace=False)]
    group_end = cluster_1.iloc[np.random.choice(cluster_1.shape[0], n, replace=False)]
    # Connect start and endpoints with a migrant for each year
    group = [np.linspace(group_start.iloc[x], group_end.iloc[x],num=22) for x in np.arange(0,n)]
    migrants = np.array(group).reshape((n,22,6))
    # Do not use start and endpoint - no migration happened yet
    migrants = migrants[:,1:21,:]
    #migrants = pd.DataFrame(group) TODO:Check which datastructure I want
    return migrants


def generate_migrants_from_distribution(n, means, stds, rand=100):
    """
    Generate a set of migrants from mean and standard deviation

    Args:
    - means (list): Mean values per cluster
    - stds (list): Standard deviation per cluster
    - n (int): number of migrants to generate

    Returns:
    - list: n migrants with 20 steps between start and end and 6 dimensions per step
    """
    data = generate_gaussian_dataset([n,n], means, stds, 6, rand)
    # Extract the datapoints which lie in each cluster and drop the predicted label
    cluster_0 = data[data['cluster'] == 0].drop(columns=['cluster'])
    cluster_1 = data[data['cluster'] == 1].drop(columns=['cluster'])
    # Connect start and endpoints with a migrant for each year
    group = [np.linspace(cluster_0.iloc[x], cluster_1.iloc[x],num=22) for x in np.arange(0,n)]
    migrants = np.array(group).reshape((n,22,6))
    # Do not use start and endpoint - no migration happened yet
    migrants = migrants[:,1:21,:]
    #migrants = pd.DataFrame(group) TODO:Check which datastructure I want
    return migrants



def line_between_two_points(point1, point2):
    straight_line_function = lambda p: point1 + p * (point2 - point1)


def calculate_thresholds_basic(data, clf, multiplier):
    """
    Calculate basis thresholds. 
    The threshold for the distance to the seperating hyperplane is the maximum of the distances from the support vector to the seperating hyperplane.
    The threshold for the distance to the cluster connecting vector is the mean plus three times the standard deviation of the distance to the vector connecting of the cluster with the higher distance standard deviation.

    Args:
    - data (Pandas dataframe): Dataset used to train our model with predicted labels
    - clf (SVM model): Model after being trained on trainingsdata
    - multiplier (float): Multiplier for the sigma rule for the distance to the CCV
    Returns:
    - t1 (float): treshold for the distance to the hyperplane
    - t2 (list): thresholds for the distance to the cluster connecting vectorr
    """
    # Extract support vectors and set maximum of distances from hyperplane as threshold
    sv = clf.support_vectors_
    t1 = np.max(linear_kernel_distance(sv,clf))#np.max([linear_kernel_distance(x, clf) for x in sv])
    # Calculate threshold based on the standard deviation of the clusters to the cluster connecting vector
    cluster_0 = data[data['cluster'] == 0].drop(columns=['cluster'])
    center_cluster_0 = cluster_0.mean(axis = 0)
    cluster_1 = data[data['cluster'] == 1].drop(columns=['cluster'])
    center_cluster_1 = cluster_1.mean(axis = 0)
    t2_c0 =  cluster_vector_distance_variance_threshold(cluster_0, center_cluster_0, center_cluster_1, multiplier)
    t2_c1 = cluster_vector_distance_variance_threshold(cluster_1, center_cluster_0, center_cluster_1, multiplier)
    t2 = [t2_c0, t2_c1]

    return t1,t2



def calculate_thresholds_basic_kernel(data, clf):
    """
    Calculate basis thresholds. 
    The threshold for the distance to the seperating hyperplane is the maximum of the distances from the support vector to the seperating hyperplane.
    The threshold for the distance to the cluster connecting vector is the mean plus three times the standard deviation of the distance to the vector connecting of the cluster with the higher distance standard deviation.

    Args:
    - data (Pandas dataframe): Dataset used to train our model with predicted labels
    - clf (SVM model): Model after being trained on trainingsdata

    Returns:
    - t1 (float): treshold for the distance to the hyperplane
    - t2 (list): thresholds for the distance to the cluster connecting vectorr
    """
    # Extract support vectors and set maximum of distances from hyperplane as threshold
    sv = clf.support_vectors_
    t1 = np.max(clf.decision_function(sv))#np.max([linear_kernel_distance(x, clf) for x in sv])
    
    # Calculate threshold based on the standard deviation of the clusters to the cluster connecting vector
    cluster_0 = data[data['cluster'] == 0].drop(columns=['cluster'])
    center_cluster_0 = cluster_0.mean(axis = 0)
    cluster_1 = data[data['cluster'] == 1].drop(columns=['cluster'])
    center_cluster_1 = cluster_1.mean(axis = 0)
    t2_c0 =  cluster_vector_distance_variance_threshold(cluster_0, center_cluster_0, center_cluster_1)
    t2_c1 = cluster_vector_distance_variance_threshold(cluster_1, center_cluster_0, center_cluster_1)
    t2 = [t2_c0, t2_c1]

    return t1,t2



def calculate_thresholds_gaussian(data, clf, multiplier):
    """
    Calculate gaussian thresholds where both thresholds are based on the distribution of distances. 
    The threshold for the distance to the seperating hyperplane is the mean minus three times the standard deviation of the distances to the seperating hyperplane.
    The threshold for the distance to the cluster connecting vector is the mean plus three times the standard deviation of the distance to the vector connecting of the cluster with the higher distance standard deviation.

    Args:
    - data (Pandas dataframe): Dataset used to train our model with predicted labels
    - clf (SVM model): Model after being trained on trainingsdata
    - multiplier (int): Multiplier for the standard deviation for the distance to the seperating hyperplane

    Returns:
    - t1 (list): thresholds for the distance to the hyperplane - first element is for the first cluster
    - t2 (list): thresholds for the distance to the cluster connecting vector - c0, c1
    """
    cluster_0 = data[data['cluster'] == 0].drop(columns=['cluster'])
    center_cluster_0 = cluster_0.mean(axis = 0)
    cluster_1 = data[data['cluster'] == 1].drop(columns=['cluster'])
    center_cluster_1 = cluster_1.mean(axis = 0)
    # Calculate Distances to seperating hyperplane for each cluster seperately
    distances_cluster_0 = linear_kernel_distance(cluster_0,clf)
    distances_cluster_1 = linear_kernel_distance(cluster_1,clf)
    # To visualize the distribution save the distances
    #dump(distances_cluster_0, 'Plots/distances_cluster_0.pkl')
    #dump(distances_cluster_1, 'Plots/distances_cluster_1.pkl')
    # Calculate std and mean for distances per cluster 
    std_c0 = np.std(distances_cluster_0)
    std_c1 = np.std(distances_cluster_1)
    mean_c0 = np.mean(distances_cluster_0)
    mean_c1 = np.mean(distances_cluster_1)
    # Calculate threshold per cluster using the three sigma rule as mean + 3*std
    # Threshold for cluster 0 is the first entry
    # Multiplier for std

    if clf.predict(np.reshape(center_cluster_0, (1, -1))) == 0:
        t1 = [mean_c0 - (multiplier*std_c0), mean_c1 - (multiplier*std_c1)]
    else:
        t1 = [mean_c1 - (multiplier*std_c1), mean_c0 - (multiplier*std_c0)]


    # Calculate threshold based on the standard deviation of the clusters to the cluster connecting vector
    t2_c0 =  cluster_vector_distance_variance_threshold(cluster_0, center_cluster_0, center_cluster_1, 3)
    t2_c1 = cluster_vector_distance_variance_threshold(cluster_1, center_cluster_0, center_cluster_1, 3)
    t2 = [t2_c0, t2_c1]

    return t1,t2

def calculate_thresholds_gaussian_kernel(data, clf, multiplier):
    """
    Calculate gaussian thresholds where both thresholds are based on the distribution of distances. 
    The threshold for the distance to the seperating hyperplane is the mean minus three times the standard deviation of the distances to the seperating hyperplane.
    The threshold for the distance to the cluster connecting vector is the mean plus three times the standard deviation of the distance to the vector connecting of the cluster with the higher distance standard deviation.

    Args:
    - data (Pandas dataframe): Dataset used to train our model with predicted labels
    - clf (SVM model): Model after being trained on trainingsdata

    Returns:
    - t1 (list): thresholds for the distance to the hyperplane - first element is for the first cluster
    - t2 (list): thresholds for the distance to the cluster connecting vector - c0, c1
    """
    cluster_0 = data[data['cluster'] == 0].drop(columns=['cluster'])
    center_cluster_0 = cluster_0.mean(axis = 0)
    cluster_1 = data[data['cluster'] == 1].drop(columns=['cluster'])
    center_cluster_1 = cluster_1.mean(axis = 0)
    # Calculate Distances to seperating hyperplane for each cluster seperately
    distances_cluster_0 = np.abs(clf.decision_function(cluster_0))
    distances_cluster_1 = np.abs(clf.decision_function(cluster_1))
    # Calculate std and mean for distances per cluster 
    std_c0 = np.std(distances_cluster_0)
    std_c1 = np.std(distances_cluster_1)
    mean_c0 = np.mean(distances_cluster_0)
    mean_c1 = np.mean(distances_cluster_1)
    # Calculate threshold per cluster using the three sigma rule as mean + 3*std
    # Threshold for cluster 0 is the first entry
    # Multiplier for std
    mult = multiplier
    if clf.predict(np.reshape(center_cluster_0, (1, -1))) == 0:
        t1 = [mean_c0 - (mult*std_c0), mean_c1 - (mult*std_c1)]
    else:
        t1 = [mean_c1 - (mult*std_c1), mean_c0 - (mult*std_c0)]


    # Calculate threshold based on the standard deviation of the clusters to the cluster connecting vector
    t2_c0 =  cluster_vector_distance_variance_threshold(cluster_0, center_cluster_0, center_cluster_1, 3)
    t2_c1 = cluster_vector_distance_variance_threshold(cluster_1, center_cluster_0, center_cluster_1, 3)
    t2 = [t2_c0, t2_c1]

    return t1,t2


def calculate_thresholds_labelled(data, clf, multiplier):
    """
    Thresholds where the cluster label of the migrant is known. For Isotope Data where the burrial site is known.
    Calculate gaussian thresholds where both thresholds are based on the distribution of distances. 
    The threshold for the distance to the seperating hyperplane is the mean minus three times the standard deviation of the distances to the seperating hyperplane.
    The threshold for the distance to the cluster connecting vector is the mean plus three times the standard deviation of the distance to the vector connecting of the cluster with the higher distance standard deviation.

    Args:
    - data (Pandas dataframe): Dataset used to train our model with predicted labels
    - clf (SVM model): Model after being trained on trainingsdata

    Returns:
    - t1 (list): mean and std for the distance to the hyperplane - first element is for cluster with predict = 0
    - t2 (list): thresholds for the distance to the cluster connecting vector - c0, c1
    """
    cluster_0 = data[data['cluster'] == 0].drop(columns=['cluster'])
    center_cluster_0 = cluster_0.mean(axis = 0)
    cluster_1 = data[data['cluster'] == 1].drop(columns=['cluster'])
    center_cluster_1 = cluster_1.mean(axis = 0)
    # Calculate Distances to seperating hyperplane for each cluster seperately
    distances_cluster_0 = np.abs(clf.decision_function(cluster_0))
    distances_cluster_1 = np.abs(clf.decision_function(cluster_1))
    # Calculate std and mean for distances per cluster 
    std_c0 = np.std(distances_cluster_0)
    std_c1 = np.std(distances_cluster_1)
    mean_c0 = np.mean(distances_cluster_0)
    mean_c1 = np.mean(distances_cluster_1)
    # Calculate threshold per cluster using the three sigma rule as mean + 3*std
    # Threshold for cluster 0 is the first entry
    # Multiplier for std
    mult = multiplier
    # TODO: CHECK IF THIS WORKS AS INTENDED
    if clf.predict(np.reshape(center_cluster_0, (1, -1))) == 0:
        t1 = [(mean_c0, std_c0), (mean_c1, std_c1)]
    else:
        t1 = [(mean_c1, mult*std_c1),  (mean_c0, std_c0)]


    # Calculate threshold based on the standard deviation of the clusters to the cluster connecting vector
    t2_c0 =  cluster_vector_distance_variance_threshold(cluster_0, center_cluster_0, center_cluster_1, 3)
    t2_c1 = cluster_vector_distance_variance_threshold(cluster_1, center_cluster_0, center_cluster_1, 3)
    t2 = [t2_c0, t2_c1]

    return t1,t2

