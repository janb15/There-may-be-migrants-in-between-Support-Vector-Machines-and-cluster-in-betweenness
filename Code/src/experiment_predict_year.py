
import numpy as np
import pandas as pd
from joblib import dump, load
from migrant_detection import predict_year
from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from utils import generate_migrants_from_distribution

STD = [[0.0023975, 0.073625 , 0.015925 , 0.13005  , 0.002775 , 0.007925 ],
       [0.001375 , 0.07545  , 0.023525 , 0.081725 , 0.0023   , 0.005625 ]]
    
MEAN = [[0.7207, 38.2853, 15.6458, 18.2447, 2.4545, 1.1645],
                          [0.70573, 39.0076, 15.8019, 18.9493, 2.4783, 1.2085]]
YEARS = [[0,5], [5,10],[10,15], [15,20]]
SEEDS = list(range(25))
results = []
# Generate Testdataset
data_columns = ['87Sr/86Sr','208Pb/204Pb','207Pb/204Pb','206Pb/204Pb','208Pb/207Pb','206Pb/207Pb']
X, label = make_blobs(n_samples=[200,200], centers=MEAN, cluster_std=STD, n_features=6, random_state=0)
data = pd.DataFrame(X, columns=data_columns)

# Scale the dataset to the interval [0,1]
scaler = MinMaxScaler() #Can be changed to robustscaler or standardscaler
data = pd.DataFrame(scaler.fit_transform(data[data_columns]), columns=data_columns)
data['cluster'] = label
cluster_0 = data[data['cluster'] == 0].drop(columns=['cluster'])
cluster_1 = data[data['cluster'] == 1].drop(columns=['cluster'])

mean_c0 = cluster_0.mean(axis=0)
mean_c1 = cluster_1.mean(axis=0)
# Set the burial label for all samples to 0
data['burial'] = 0
# Generate Migrants
n = 100000
migrants = generate_migrants_from_distribution(n, MEAN, STD, rand=0)
# Scale Migrants
migrants_scaled = scaler.transform(migrants.reshape(-1, migrants.shape[-1]))
# Predict Years
year_predictions = np.array([predict_year(mean_c0,mean_c1, x, 1) for x in migrants_scaled])
# Create true label
sequence = np.arange(1, 21)  # Create the sequence 1 to 20
true_label = np.tile(sequence, n)  # Repeat the sequence n times
# Calculate Metrics
mse = mean_squared_error(true_label, year_predictions)
rmse = mean_squared_error(true_label, year_predictions, squared=False)
print("Mean Squared Error = " + str(mse))
print("Root Mean Squared Error = " + str(rmse))

# For Groups of Years
"""
for year in YEARS:
    # Extract migrants from a specific range of years
    migrants_y = migrants[:,year[0]:year[1],:] # groups of years
    migrants_scaled = scaler.transform(migrants_y.reshape(-1, migrants_y.shape[-1]))
    year_predictions = [predict_year(mean_c0,mean_c1, x, 1) for x in migrants_scaled]

    year_predictions = np.array(year_predictions)
    print("N = " + str(len(year_predictions)))
    pred_bins = np.where((year_predictions > year[0]) & (year_predictions <= year[1]), 1, 0)

    #print(true_label)
    score = f1_score(np.ones((len(year_predictions),1)), pred_bins)
 
    #accuracy = accuracy_score(true_label, year_predictions)
    #recall = recall_score(true_label, year_predictions, average=None)
    #precision = precision_score(true_label, year_predictions, average=None)
    #print("Accuracy  = " + str(accuracy))
    #print("precision= " + str(precision))
    #print("recall = " + str(recall))
    print("F1-Score = " + str(score))
"""