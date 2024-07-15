import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
from migrant_detection import predict_migrant_linear_dynamic_threshold
from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from utils import (calculate_thresholds_gaussian,
                   generate_migrants_from_distribution)

from datetime import date

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

# Experiment on the impact of unbalanced classes
STD = [[0.0023975, 0.073625 , 0.015925 , 0.13005  , 0.002775 , 0.007925 ],
       [0.001375 , 0.07545  , 0.023525 , 0.081725 , 0.0023   , 0.005625 ]]
    
MEAN = [[0.7207, 38.2853, 15.6458, 18.2447, 2.4545, 1.1645],
                          [0.70573, 39.0076, 15.8019, 18.9493, 2.4783, 1.2085]]

SAMPLES_NUMBERS = [25,50,100,200,500,1000,10000,25000, 50000, 75000, 100000,150000, 250000]
SEEDS = list(range(25))
results = []
# Generate Testdataset

for sample_number in SAMPLES_NUMBERS:
    
    X, label = make_blobs(n_samples=[200,sample_number], centers=MEAN, cluster_std=STD, n_features=6, random_state=0)
    data = pd.DataFrame(X, columns=['87Sr/86Sr','208Pb/204Pb','207Pb/204Pb','206Pb/204Pb','208Pb/207Pb','206Pb/207Pb'])
    data['cluster'] = label
    # Extract Variable Names without cluster label
    data_columns = data.columns.to_list()[:-1]
    # Scale the dataset to the interval [0,1]
    scaler = MinMaxScaler() #Can be changed to robustscaler or standardscaler

    # Train the SVM Model and scale the test and train set
    data_train, data_test, cluster_train, cluster_test = train_test_split(data[data.columns[:-1]], data['cluster'], random_state=0)
    data_train_scaled = pd.DataFrame(scaler.fit_transform(data_train), columns=data_columns)
    data_test_scaled = pd.DataFrame(scaler.transform(data_test), columns=data_columns)
    model = SVC(kernel='linear', C = 10E10, random_state=0, cache_size=4000)
    model.fit(data_train_scaled, cluster_train)
    for s in SEEDS:
        # Generate migrant samples - similar amount than test set
        n = 1 if len(data_test_scaled.values)//20 == 0 else len(data_test_scaled.values)//20
        migrants = generate_migrants_from_distribution(n, MEAN, STD, rand=s)
        # Scale migrants
        migrants_scaled = scaler.transform(migrants.reshape(-1, migrants.shape[-1]))
        data_train_scaled['cluster'] = cluster_train.values

        # Generate basic threshold for distance to hyperplane(t1) and distance to cluster connecting vector(t2)
        t1, t2 = calculate_thresholds_gaussian(data_train_scaled, model, 2)

        # Drop labels
        data_train_scaled = data_train_scaled.drop(columns=['cluster'])

        predictions_migrant = predict_migrant_linear_dynamic_threshold(migrants_scaled, data_train_scaled, model, t1, t2)
        predictions_data_test = predict_migrant_linear_dynamic_threshold(data_test_scaled.to_numpy(), data_train_scaled, model, t1, t2)
        # Calculate Score
        # Generated Migrants should be labelled as Migrant and the test_data(Datapoints from both clusters not used in model training) as non-migrant

        # Generate true labels
        true_label = np.concatenate((np.ones((len(predictions_migrant),1)),np.zeros((len(predictions_data_test),1))), axis=0)
        pred_label = np.concatenate((np.array(predictions_migrant), np.array(predictions_data_test)), axis=0)
        score = f1_score (true_label, pred_label)
        accuracy = accuracy_score(true_label, pred_label)
        recall = recall_score(true_label, pred_label)
        precision = precision_score(true_label, pred_label)
        #print("Accuracy  = " + str(accuracy))
        #print("precision= " + str(precision))
        #print("recall = " + str(recall))
        #print("F1-Score = " + str(score))
        results.append([s,sample_number,  accuracy, recall, precision, score])
        # Calculate F1-Score https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
results = pd.DataFrame(results, columns=['Seed', 'Samples', 'Accuracy', 'Recall', 'Precision', 'Score'])
dump(results, 'experiment_result/experiment03_unbalanced_clusters.pkl')