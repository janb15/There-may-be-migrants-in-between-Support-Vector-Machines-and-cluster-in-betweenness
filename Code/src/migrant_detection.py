import numpy as np


def distance_to_plane(point1, point2, target_point):
    # Math Source: https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    # https://paulbourke.net/geometry/pointlineplane/
    # https://softwareengineering.stackexchange.com/questions/168572/distance-from-point-to-n-dimensional-line
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line

    # Calculate the vector between the two plane-defining points
    line_vector = point2 - point1

    # Calculate the vector from point1 to the target point
    point1_to_target = target_point - point1

    # Calculate the distance from the point to the plane
    t = np.dot(point1_to_target,line_vector)/np.dot(line_vector,line_vector)
    distance = np.linalg.norm(point1_to_target - t * line_vector)
    
    return distance


def adaptive_threshold(t2_c0,t2_c1,c0,c1,x):
    """
    Calculate the adaptive threshold for distance to cluster connecting vector

    Args:
    - t2_c0 (float): Threshold for cluster 0
    - t2_c1 (float): Threshold for cluster 1
    - c0 (float): Center for cluster 0
    - c1 (float): Center for cluster 1
    - x (list): data point to label

    Returns:
    - threshold (float): adaptive threshold for distance to cluster connecting vector
    """
    
    # Calculate vector from one cluster center to data point x
    x_c0 = x - c0
    # Calculate cluster connecting vector
    cluster_vector = c0 -  c1
    # Project vector from cluster center to data point onto cluster connecting vector
    proj = np.dot(x_c0,cluster_vector)/np.dot(cluster_vector,cluster_vector)*cluster_vector
    # Calculate the fraction describing how far we travelled along the cluster connecting vector
    frac = np.linalg.norm(proj)/np.linalg.norm(cluster_vector)
    # Calculate the dynamic threshold based on how far we are along the cluster connecting vector
    # If it is behind one of the centers set the threshold to -inf

    # Check if within bounds
    # Reverse computation
    x_c1 = x - c1
    proj_r = np.dot(x_c1,cluster_vector)/np.dot(cluster_vector,cluster_vector)*cluster_vector
    frac_r = np.linalg.norm(proj_r)/np.linalg.norm(cluster_vector)
    in_bounds = frac < 1 and frac_r < 1
    # Check if behind clusters
    if in_bounds:
        threshold =  t2_c0 + ((t2_c1-t2_c0)* frac) 
    else:
        threshold =  float('-inf')
    return threshold 


def predict_year(c0,c1,x, burial_x):
    """
    Predict the years after migration for sample x. We assume a full adaption to the new isotope signal after 20 years.

    Args:
    - c0 (float): Center for cluster 0
    - c1 (float): Center for cluster 1
    - x (list): data point to label
    - burial_x: burial label of sample x

    Returns:
    - year (int): predicted years that passed since migration
    """
    # For migrants found in cluster 1
    if burial_x == 1:
        # Calculate vector from one cluster center to data point x
        x_c0 = x - c0
        # Calculate cluster connecting vector
        cluster_vector = c0 -  c1
        # Project vector from cluster center to data point onto cluster connecting vector
        proj = np.dot(x_c0,cluster_vector)/np.dot(cluster_vector,cluster_vector)*cluster_vector
        # Calculate the fraction describing how far we travelled along the cluster connecting vector
        frac = np.linalg.norm(proj)/np.linalg.norm(cluster_vector)
    else:
        # For migrants found in cluster 0
        x_c1 = x - c1
        cluster_vector = c0 -  c1
        proj = np.dot(x_c1,cluster_vector)/np.dot(cluster_vector,cluster_vector)*cluster_vector
        frac = np.linalg.norm(proj)/np.linalg.norm(cluster_vector)
    return 20*frac 

def linear_kernel_distance(x, clf):
    """
    Calculate the distance to the decision boundary when using a linear kernel

    Args:
    - x (list): List of data points
    - clf : SVM model

    Returns:
    - list of distances
    """
    y = np.array(clf.decision_function(x))
    w_norm = np.linalg.norm(clf.coef_)
    dist = y/w_norm
    result = [abs(v) for v in dist]
    return result


def check_thresholds(Prediction, A, B, t1, t2):
    """
    Check if each element of A is smaller than t1 and the corresponding element of B is smaller than t2.

    Args:
    - Prediction: predictions or burials of samples
    - A (list): List of numbers
    - B (list): List of numbers
    - t1 (float): Threshold for vector A
    - t2 (float): Threshold for vector B

    Returns:
    - list of booleans: result[i] true if within both thresholds else false
    """
    # Check if the lengths of vectors A and B are the same
    if len(A) != len(B):
        raise ValueError("Vectors A and B must have the same length")
    result = []
    if  isinstance(t1, list):
        for i in range(len(A)):
            if Prediction[i] >= 0: # if the data point is a positive instance use the first threshold in t1
                result.append(int(A[i] <= t1[0] and B[i] <= t2))
            else:# if the data point is a negative instance use the second threshold in t1
                result.append(int(A[i] <= t1[1] and B[i] <= t2))
        return result
        
    else:
        # Check if A[i] < t1 and B[i] < t2 for all i
        for i in range(len(A)):
            result.append(int(A[i] <= t1 and B[i] <= t2))
        return result



def predict_migrant_linear(x, data, clf, t1, t2):
    """
    Check if a point is a migrant when applying the approach based on linear SVMs

    Args:
    - x (list): Datapoint
    - Trainingdata (Pandas dataframe): Dataset used to train our model
    - clf (SVM model): Model after being trained on trainingsdata
    - t1 (float): Threshold for distance to seperating hyperplane
    - t2 (float): Threshold for distance to connecting vector

    Returns:
    - list of booleans: result[i] true if within both thresholds else false
    """
    # calculate the distance to the hyperplane of the SVM modelLabel 
    distances_from_hyperplane = linear_kernel_distance(x, clf)
    # calculate Clustercenters
    data_with_prediction = data.copy()
    data_with_prediction['Prediction'] = clf.predict(data_with_prediction)
    cluster0 = data_with_prediction[data_with_prediction['Prediction'] == 0]
    cluster0 = cluster0.drop(columns=['Prediction'])
    cluster1 = data_with_prediction[data_with_prediction['Prediction'] == 1]
    cluster1 = cluster1.drop(columns=['Prediction'])
    #data = data.drop(columns=['Prediction'])
    representant0,representant1 = cluster0.mean(axis=0), cluster1.mean(axis=0)
    # calculate the distance to the cluster connecting vector
    distance_from_connecting_vector = [distance_to_plane(representant0.values,representant1.values,x1) for x1 in x]
    # Check if distances are within threshold
    result = check_thresholds(clf.predict(x), distances_from_hyperplane, distance_from_connecting_vector, t1, t2)
    return result

def predict_migrant_linear_dynamic_threshold(x, data, clf, t1, t2):
    """
    Check if a point is a migrant when applying the approach based on linear SVMs

    Args:
    - x (list): List of Datapoints
    - Trainingdata (Pandas dataframe): Dataset used to train our model
    - clf (SVM model): Model after being trained on trainingsdata
    - t1 (float): Threshold for distance to seperating hyperplane
    - t2 (float): Threshold for distance to connecting vector

    Returns:
    - list of booleans: result[i] true if within both thresholds else false
    """
    # calculate the distance to the hyperplane of the SVM modelLabel 
    distances_from_hyperplane = linear_kernel_distance(x, clf)
    # calculate Clustercenters
    data_with_prediction = data.copy()
    data_with_prediction['Prediction'] = clf.predict(data_with_prediction)
    cluster0 = data_with_prediction[data_with_prediction['Prediction'] == 0]
    cluster0 = cluster0.drop(columns=['Prediction'])
    cluster1 = data_with_prediction[data_with_prediction['Prediction'] == 1].drop(columns=['Prediction'])
    #data = data.drop(columns=['Prediction'])
    representant0,representant1 = cluster0.mean(axis=0), cluster1.mean(axis=0)
    # calculate the distance to the cluster connecting vector
    distance_from_connecting_vector = [distance_to_plane(representant0.values,representant1.values,x1) for x1 in x]
    # calculate dynamic threshold for cluster connecting vector
    center_cluster_0 = cluster0.mean(axis = 0)
    center_cluster_1 = cluster1.mean(axis = 0)
    # Calculate predictions for x
    predictions = clf.predict(x)
    result = []
    if  isinstance(t1, list):
        for i in range(len(x)):
            # calculate dynamic threshold for cluster connecting vector
            dynamic_t2 = adaptive_threshold(t2[0],t2[1],center_cluster_0,center_cluster_1,x[i])
            if predictions[i] == 0: # if the data point is an instance on the side of the hyperplane of cluster 0 use the first threshold in t1
                # Check thresholds
                result.append(int(distances_from_hyperplane[i] <= t1[0] and distance_from_connecting_vector[i] <= dynamic_t2))
            else:# if the data point is an instance on the side of the hyperplane of cluster 1 use the second threshold in t1
                result.append(int(distances_from_hyperplane[i] <= t1[1] and distance_from_connecting_vector[i] <= dynamic_t2))
        
    else:
        # Check if A[i] < t1 and B[i] < t2 for all i
        for i in range(len(x)):
            dynamic_t2 = adaptive_threshold(t2[0],t2[1],center_cluster_0,center_cluster_1,x[i])
            result.append(int(distances_from_hyperplane[i] <= t1 and distance_from_connecting_vector[i] <= dynamic_t2))
    return result


def predict_migrant_kernel(x, data, clf, t1, t2):
    """
    Check if a point is a migrant when applying the approach based on linear SVMs

    Args:
    - x (list): List of Datapoints
    - Trainingdata (Pandas dataframe): Dataset used to train our model
    - clf (SVM model): Model after being trained on trainingsdata
    - t1 (float): Threshold for distance to seperating hyperplane
    - t2 (float): Threshold for distance to connecting vector

    Returns:
    - list of booleans: result[i] true if within both thresholds else false
    """
    # calculate the distance to the hyperplane of the SVM modelLabel 
    distances_from_hyperplane = np.abs(clf.decision_function(x))
    # calculate Clustercenters
    data_with_prediction = data.copy()
    data_with_prediction['Prediction'] = clf.predict(data_with_prediction)
    cluster0 = data_with_prediction[data_with_prediction['Prediction'] == 0]
    cluster0 = cluster0.drop(columns=['Prediction'])
    cluster1 = data_with_prediction[data_with_prediction['Prediction'] == 1].drop(columns=['Prediction'])
    #data = data.drop(columns=['Prediction'])
    representant0,representant1 = cluster0.mean(axis=0), cluster1.mean(axis=0)
    # calculate the distance to the cluster connecting vector
    distance_from_connecting_vector = [distance_to_plane(representant0.values,representant1.values,x1) for x1 in x]
    # calculate dynamic threshold for cluster connecting vector
    center_cluster_0 = cluster0.mean(axis = 0)
    center_cluster_1 = cluster1.mean(axis = 0)
    # Calculate predictions for x
    predictions = clf.predict(x)
    result = []
    if  isinstance(t1, list):
        for i in range(len(x)):
            # calculate dynamic threshold for cluster connecting vector
            dynamic_t2 = adaptive_threshold(t2[0],t2[1],center_cluster_0,center_cluster_1,x[i])
            if predictions[i] == 0: # if the data point is an instance on the side of the hyperplane of cluster 0 use the first threshold in t1
                # Check thresholds
                result.append(int(distances_from_hyperplane[i] <= t1[0] and distance_from_connecting_vector[i] <= dynamic_t2))
            else:# if the data point is an instance on the side of the hyperplane of cluster 1 use the second threshold in t1
                result.append(int(distances_from_hyperplane[i] <= t1[1] and distance_from_connecting_vector[i] <= dynamic_t2))
        
    else:
        # Check if A[i] < t1 and B[i] < t2 for all i
        for i in range(len(x)):
            dynamic_t2 = adaptive_threshold(t2[0],t2[1],center_cluster_0,center_cluster_1,x[i])
            result.append(int(distances_from_hyperplane[i] <= t1 and distance_from_connecting_vector[i] <= dynamic_t2))
    return result

def predict_migrant_kernel_labelled(x, x_label, data, clf, t1, t2, multiplier):
    """
    Check if a point is a migrant when applying the approach based on linear SVMs

    Args:
    - x (list): List of Datapoints
    - x_label(list): Label of origin for each data sample which is used to adapt the threshold based on location
    - Trainingdata (Pandas dataframe): Dataset used to train our model
    - clf (SVM model): Model after being trained on trainingsdata
    - t1 (float): Threshold for distance to seperating hyperplane
    - t2 (float): Threshold for distance to connecting vector
    - multiplier (float): Multiplier for standard deviation of distances to hyperplane

    Returns:
    - list of booleans: result[i] true if within both thresholds else false
    """
    # calculate the distance to the hyperplane of the SVM model
    distances_from_hyperplane = np.abs(clf.decision_function(x))
    # calculate Clustercenters
    data_with_prediction = data.copy()
    data_with_prediction['Prediction'] = clf.predict(data_with_prediction)
    cluster0 = data_with_prediction[data_with_prediction['Prediction'] == 0]
    cluster0 = cluster0.drop(columns=['Prediction'])
    cluster1 = data_with_prediction[data_with_prediction['Prediction'] == 1].drop(columns=['Prediction'])
    #data = data.drop(columns=['Prediction'])
    representant0,representant1 = cluster0.mean(axis=0), cluster1.mean(axis=0)
    # calculate the distance to the cluster connecting vector
    distance_from_connecting_vector = np.array([distance_to_plane(representant0.values,representant1.values,x1) for x1 in x])
    # calculate dynamic threshold for cluster connecting vector
    center_cluster_0 = cluster0.mean(axis = 0)
    center_cluster_1 = cluster1.mean(axis = 0)
    # Calculate predictions for x
    predictions = clf.predict(x)
    result = []
    if  isinstance(t1, list):
        for i in range(len(x)):
            # calculate dynamic threshold for cluster connecting vector
            dynamic_t2 = adaptive_threshold(t2[0],t2[1],center_cluster_0,center_cluster_1,x[i])
            if predictions[i] == x_label[i]: # Check if the datapoint lies on the side of the hyperplane of the cluster it is labelled as    
                if predictions[i] == 0: # if the data point is an instance on the side of the hyperplane of cluster 0 use the first threshold in t1
                    # Check thresholds
                    result.append(int(distances_from_hyperplane[i] <= t1[0][0] - (t1[0][1] * multiplier)  and distance_from_connecting_vector[i] <= dynamic_t2))
                else:# if the data point is an instance on the side of the hyperplane of cluster 1 use the second threshold in t1
                    result.append(int(distances_from_hyperplane[i] <= t1[1][0] - (t1[1][1] * multiplier) and distance_from_connecting_vector[i] <= dynamic_t2))
            else: # If the datapoint is on the other side we can use a larger threshold
                if predictions[i] == 0: # if the data point is an instance on the side of the hyperplane of cluster 0 use the first threshold in t1
                    # Check thresholds
                    result.append(int(distances_from_hyperplane[i] <= t1[0][0] + (t1[0][1] * multiplier)  and distance_from_connecting_vector[i] <= dynamic_t2))
                else:# if the data point is an instance on the side of the hyperplane of cluster 1 use the second threshold in t1
                    result.append(int(distances_from_hyperplane[i] <= t1[1][0] + (t1[1][1] * multiplier) and distance_from_connecting_vector[i] <= dynamic_t2))
    else:
        raise SystemExit("Wrong threshold calculation applied")
    return result


