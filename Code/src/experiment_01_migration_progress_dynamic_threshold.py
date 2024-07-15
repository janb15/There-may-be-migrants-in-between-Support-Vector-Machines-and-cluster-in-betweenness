import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
from migrant_detection import (
                               predict_migrant_linear_dynamic_threshold)
# Mahalanobis Distance
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.datasets import make_blobs
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from utils import (calculate_thresholds_basic, calculate_thresholds_gaussian,
                   generate_migrants_from_distribution)

# Experiment on the impact of the year of migration 
# One experiment for each single year -> look for "per year" comments
# One experiment for groups of years -> look for "groups of years" comments
YEARS = [[1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7], [8,8], [9,9], [10,10], [11,11], [12,12], [13,13], [14,14], [15,15], [16,16], [17,17], [18,18], [19,19], [20,20], ] # Per year
#YEARS = [[1,4], [5,8], [9,12], [13,16], [16,20]] # groups of years
#YEARS = [[1,20]] # all years
STD = [[0.0023975, 0.073625 , 0.015925 , 0.13005  , 0.002775 , 0.007925 ],
       [0.001375 , 0.07545  , 0.023525 , 0.081725 , 0.0023   , 0.005625 ]]
    
MEAN = [[0.7207, 38.2853, 15.6458, 18.2447, 2.4545, 1.1645],
                          [0.70573, 39.0076, 15.8019, 18.9493, 2.4783, 1.2085]]
SEEDS = list(range(100))
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

# Train the SVM Model and scale the test and train set
data_train, data_test, cluster_train, cluster_test = train_test_split(data[data.columns[:-1]], data['cluster'], random_state=0)
data_train_scaled = pd.DataFrame(scaler.fit_transform(data_train), columns=data_columns)
data_test_scaled = pd.DataFrame(scaler.transform(data_test), columns=data_columns)
model = SVC(kernel='linear', C = 10E10, random_state=0, cache_size=4000)
model.fit(data_train_scaled, cluster_train)

# Generate threshold for distance to hyperplane(t1) and distance to cluster connecting vector(t2)
data_train_scaled['cluster'] = cluster_train.values

t1, t2 = calculate_thresholds_gaussian(data_train_scaled, model, 2)
# Select clusters
cluster_0 = data_train_scaled[data_train_scaled['cluster'] == 0]
cluster_0 = cluster_0.drop(columns=['cluster'])
cluster_1 = data_train_scaled[data_train_scaled['cluster'] == 1]
cluster_1 = cluster_1.drop(columns=['cluster'])
# fit a MCD robust estimator to data
robust_cov_c0 = MinCovDet().fit(cluster_0)
robust_cov_c1 = MinCovDet().fit(cluster_1)
# fit a MLE estimator to data
# emp_cov = EmpiricalCovariance().fit(data_train_scaled)

# Drop labels
data_train_scaled = data_train_scaled.drop(columns=['cluster'])

# Visualize Confusionmatrix
#predictions = model.predict(data_test_scaled) # 
#cm = confusion_matrix(cluster_test, predictions, labels=model.classes_)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
#disp.plot()
#plt.show()
# Only to visualize the scaling
#data_scaled_plot = pd.DataFrame(np.concatenate((data_train_scaled,data_test_scaled)),columns=data_columns)
#sns.pairplot(data_scaled_plot)
#plt.show()

results = []


for s in SEEDS:
    # Generate 10 migrants and a datapoint for each year since the migration
    n=100 # per year
    #n = 25# groups of years
    #n = 5 # All years
    migrants_all_years = generate_migrants_from_distribution(n, MEAN, STD, rand=s)
    for year in YEARS:
        # Extract migrants from a specific range of years
        migrants = migrants_all_years[:,year[0]-1:year[1],:] # groups of years
        #migrants = migrants_all_years[:,year-1:,:] #per year


        migrants_scaled = scaler.transform(migrants.reshape(-1, migrants.shape[-1]))


        # Predict migrants
        predictions_migrant = predict_migrant_linear_dynamic_threshold(migrants_scaled, data_train_scaled, model, t1, t2)
        predictions_data_test = predict_migrant_linear_dynamic_threshold(data_test_scaled.to_numpy(), data_train_scaled, model, t1, t2)
        
        # Calculate Mahalanobis Distance
        m_dist_c0 = robust_cov_c0.mahalanobis(migrants_scaled)
        m_dist_c1 =robust_cov_c1.mahalanobis(migrants_scaled)
        m_dist = [min(x,y) for x,y in zip(m_dist_c0,m_dist_c1)]
        
        robust_cov_c1.dist_
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
        results.append([s, year[0], year[1], predictions_migrant, m_dist_c0, m_dist_c1, m_dist, robust_cov_c0.dist_, robust_cov_c1.dist_, accuracy, recall, precision, score])
        # Calculate F1-Score https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

        """  # Plot
        # Extract the first two dimensions for plotting
        x = migrants_scaled[:, 0].flatten()
        y = migrants_scaled[:, 1].flatten()
       # data_train_scaled = pd.DataFrame(data_train_scaled, columns=data_columns)

        x1 = data_train_scaled['87Sr/86Sr']
        y1 = data_train_scaled['208Pb/204Pb']
        plt.scatter(x1,y1)
        # # Create an array of colors based on the year (from 0 to 19)
        colors = np.tile(np.arange(20), n)
        # Plot detected migrants
        plt.scatter(x, y, c=predictions_migrant) # Color based on predicted label
        sns.scatterplot() # Color based on year
        #plt.scatter(data_test_scaled['87Sr/86Sr'], data_test_scaled['208Pb/204Pb'], c=predictions_data_test) # TODO: In case all neg or pos -> change color

        # Plot support vectors
        support = np.array([[x[0],x[1], 3] for x in model.support_vectors_]) #np.append(np.array(clf.support_vectors_),[[3],[3]],axis=1)
        plt.scatter(support[:,0], support[:,1], c='red')
        plt.show()
        #print('debug')
        #plot.scatter """
results = pd.DataFrame(results, columns=['Seed', 'Start_Year', 'End_Year', 'Migrant_Prediction' ,'Mahalanobis_migrant_c0', 'Mahalanobis_migrant_c1', 'Min_Mahalanobis', 'Mahalanobis_c0', 'Mahalanobis_c1', 'Accuracy', 'Recall', 'Precision', 'Score'])
dump(results, 'experiment_result/experiment01_per_years_seeds_multiplier_2.pkl')