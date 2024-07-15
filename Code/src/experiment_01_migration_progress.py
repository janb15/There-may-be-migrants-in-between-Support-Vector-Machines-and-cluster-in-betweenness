import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
from migrant_detection import predict_migrant_linear
from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from utils import (calculate_thresholds_basic,
                   generate_migrants_from_distribution)

from datetime import date

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

# Experiment on the impact of the year of migration 
# One experiment for each single year -> look for "per year comments"
# One experiment for groups of years -> look for "groups of years"
#YEARS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] # Per year
#YEARS = [[1,4], [5,8], [9,12], [13,16], [16,20]] # groups of years
YEARS = [[1,20]] # all years
STD = [[0.0023975, 0.073625 , 0.015925 , 0.13005  , 0.002775 , 0.007925 ],
       [0.001375 , 0.07545  , 0.023525 , 0.081725 , 0.0023   , 0.005625 ]]
    
MEAN = [[0.7207, 38.2853, 15.6458, 18.2447, 2.4545, 1.1645],
                          [0.70573, 39.0076, 15.8019, 18.9493, 2.4783, 1.2085]]
# Generate Testdataset

X, label = make_blobs(n_samples=[200,200], centers=MEAN, cluster_std=STD, n_features=6, random_state=0)
data = pd.DataFrame(X, columns=['87Sr/86Sr','208Pb/204Pb','207Pb/204Pb','206Pb/204Pb','208Pb/207Pb','206Pb/207Pb'])
data['cluster'] = label

# Extract Variable Names without cluster label
data_columns = data.columns.to_list()[:-1]

# Plot Data before Scaling
#sns.pairplot(data, hue="cluster")#
#plt.show()
# Scale the dataset to the interval [0,1]
scaler = MinMaxScaler() #Can be changed to robustscaler or standardscaler

# Plot Data after Scaling


# Train the SVM Model and scale the test and train set
data_train, data_test, cluster_train, cluster_test = train_test_split(data[data.columns[:-1]], data['cluster'], random_state=0)
data_train_scaled = pd.DataFrame(scaler.fit_transform(data_train), columns=data_columns)
data_test_scaled = pd.DataFrame(scaler.transform(data_test), columns=data_columns)
model = SVC(kernel='linear', C = 10E10, random_state=0, cache_size=4000)
model.fit(data_train_scaled, cluster_train)


results = []
for year in YEARS:
    #n=100 # per year
    #n = 25# groups of years
    n = 5# All years
    migrants = generate_migrants_from_distribution(n, MEAN, STD, rand=24) 

    # Extract migrants from a specific range of years
    migrants = migrants[:,year[0]-1:year[1],:] # groups of years
    #migrants = migrants[:,year-1:,:] #per year


    migrants_scaled = scaler.transform(migrants.reshape(-1, migrants.shape[-1]))
    # Plot data after scaling
    """
    # Extract the first two dimensions for plotting
    x = migrants_scaled[:, 0].flatten()
    y = migrants_scaled[:, 1].flatten()
    # Probably delete cluster_0 and cluster_1 - following 4 rows
    #data_train_scaled = pd.DataFrame(data_train_scaled, columns=data_columns)
    data_train_scaled['cluster'] = cluster_train.values
    cluster_0 = data_train_scaled[data_train_scaled['cluster'] == 0].drop(columns=['cluster'])
    cluster_1 = data_train_scaled[data_train_scaled['cluster'] == 1].drop(columns=['cluster'])
    x1 = data_train_scaled['87Sr/86Sr']
    y1 = data_train_scaled['208Pb/204Pb']
    plt.scatter(x1,y1)
    """
    # Generate basic threshold for distance to hyperplane(t1) and distance to cluster connecting vector(t2)
    t1, t2 = calculate_thresholds_basic(data_train_scaled, model)
    t2 = np.max(t2)

    # Drop labels
    data_train_scaled = data_train_scaled.drop(columns=['cluster'])

    predictions_migrant = predict_migrant_linear(migrants_scaled, data_train_scaled, model, t1, t2)
    predictions_data_test = predict_migrant_linear(data_test_scaled.to_numpy(), data_train_scaled, model, t1, t2)


    # Calculate Score
    # Generated Migrants should be labelled as Migrant and the test_data(Datapoints from both clusters not used in model training) as non-migrant

    # Generate true labels
    true_label = np.concatenate((np.ones((len(predictions_migrant),1)),np.zeros((len(predictions_data_test),1))), axis=0)
    pred_label = np.concatenate((np.array(predictions_migrant), np.array(predictions_data_test)), axis=0)
    score = f1_score (true_label, pred_label)
    accuracy = accuracy_score(true_label, pred_label)
    recall = recall_score(true_label, pred_label)
    precision = precision_score(true_label, pred_label)
    print("Accuracy  = " + str(accuracy))
    print("precision= " + str(precision))
    print("recall = " + str(recall))
    print("F1-Score = " + str(score))
    results.append([accuracy, recall, precision, score])
    # Calculate F1-Score https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

#dump(results, 'experiment_result/experiment01_groups_of_years.pkl')