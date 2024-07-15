import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
from utils import generate_migrants_from_distribution
from utils import calculate_thresholds_gaussian
from utils import overlap_check
from train import train_lin_simple
from migrant_detection import predict_migrant_linear_dynamic_threshold
from data_generation import generate_testdata
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from datetime import date
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Experiment on the impact of the year of migration 
# One experiment for each single year -> look for "per year comments"
# One experiment for groups of years -> look for "groups of years"
#YEARS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] # Per year
STD = np.array([[0.0023975, 0.073625 , 0.015925 , 0.13005  , 0.002775 , 0.007925 ],
       [0.001375 , 0.07545  , 0.023525 , 0.081725 , 0.0023   , 0.005625 ]])
    
MEAN = [[0.7207, 38.2853, 15.6458, 18.2447, 2.4545, 1.1645],
                          [0.70573, 39.0076, 15.8019, 18.9493, 2.4783, 1.2085]]
STD_MULTIPLIERS_C0 = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0]#list(np.arange(1,3.1,0.1))#[1.5,2,3,4,5]#[[1.5,1], [2,1], [3,1], [4,1], [5,1]]
STD_MULTIPLIERS_C1 = [1]#[1, 0.8, 0.6, 0.4, 0.2]
SEEDS = list(range(100))
# STD pro dimension erhöhen

results = []
# Generate Testdataset
for multiplier in STD_MULTIPLIERS_C0:
  STD = np.array([[0.0023975, 0.073625 , 0.015925 , 0.13005  , 0.002775 , 0.007925 ],
       [0.001375 , 0.07545  , 0.023525 , 0.081725 , 0.0023   , 0.005625 ]])
  STD = [STD[0]*multiplier, STD[1]]
  X, label = make_blobs(n_samples=[200,200], centers=MEAN, cluster_std=STD, n_features=6, random_state=0)
  data = pd.DataFrame(X, columns=['87Sr/86Sr','208Pb/204Pb','207Pb/204Pb','206Pb/204Pb','208Pb/207Pb','206Pb/207Pb'])
  data['cluster'] = label

  cluster_0 = data[data['cluster'] == 0].drop(columns=['cluster'])
  cluster_1 = data[data['cluster'] == 1].drop(columns=['cluster'])
  covariance_matrix_cluster_0 = np.cov(cluster_0,rowvar=0)
  covariance_matrix_cluster_1 = np.cov(cluster_1,rowvar=0)
  trace_cluster_0 = np.trace(covariance_matrix_cluster_0)
  trace_cluster_1 = np.trace(covariance_matrix_cluster_1)
  det_cluster_0 = np.linalg.det(covariance_matrix_cluster_0)
  det_cluster_1 = np.linalg.det(covariance_matrix_cluster_1)
  print("Cluster 0 trace = " + str(trace_cluster_0) + " Cluster 0 determinant = " + str(det_cluster_0))
  print("Cluster 1 trace = " + str(trace_cluster_1) + " Cluster 1 determinant = " + str(det_cluster_1))
  overlaps = np.array(overlap_check(data))
  print("Multiplier = " + str(multiplier))
  if  any(overlaps) == True:
    print("Clusters overlap")
    #exit()
  num_overlap = overlaps.sum()
  # Extract Variable Names without cluster label
  data_columns = data.columns.to_list()[:-1]
  # Scale the dataset to the interval [0,1]
  scaler = MinMaxScaler() #Can be changed to robustscaler or standardscaler
  #data_scaled = scaler.fit_transform(data[data.columns[:-1]].to_numpy())
  #data_scaled = pd.DataFrame(data_scaled , columns=data_columns)
  #data_scaled['cluster'] = label
  #dump(data_scaled, "/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/input/scaled.pkl") #Debugging

  # Train the SVM Model and scale the test and train sets
  data_train, data_test, cluster_train, cluster_test = train_test_split(data[data.columns[:-1]], data['cluster'], random_state=0)
  data_train_scaled = pd.DataFrame(scaler.fit_transform(data_train), columns=data_columns)
  data_test_scaled = pd.DataFrame(scaler.transform(data_test), columns=data_columns)
  model = SVC(kernel='linear', C = 10E10, random_state=0, cache_size=4000)
  model.fit(data_train_scaled, cluster_train)
  predictions = model.predict(data_test_scaled) # TODO: Visualisierung hier wahrscheinlich entfernen
  cm = confusion_matrix(cluster_test, predictions, labels=model.classes_)
  score_svm = f1_score (cluster_test, predictions)
  

  #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
  #disp.plot()
  #plt.show()
  # Only to visualize the scaling
  #data_scaled_plot = pd.DataFrame(np.concatenate((data_train_scaled,data_test_scaled)),columns=data_columns)
  #sns.pairplot(data_scaled_plot)
  #plt.show()
   # Generate threshold for distance to hyperplane(t1) and distance to cluster connecting vector(t2)
  data_train_scaled['cluster'] = cluster_train.values
  t1, t2 = calculate_thresholds_gaussian(data_train_scaled, model, 2)
  # Drop labels
  data_train_scaled = data_train_scaled.drop(columns=['cluster'])
  for s in SEEDS:
    # Generate migrant samples - similar amount than test set
    n = 1 if len(data_test_scaled.values)//20 == 0 else len(data_test_scaled.values)//20   # groups of years
    migrants = generate_migrants_from_distribution(n, MEAN, STD, rand=s)#, rand=24) # TODO Mehr Seeds nutzen und dann Mittelwert und STD 
    # Scale migrants
    migrants_scaled = scaler.transform(migrants.reshape(-1, migrants.shape[-1]))

  

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
    print("Accuracy  = " + str(accuracy))
    print("precision= " + str(precision))
    print("recall = " + str(recall))
    print("F1-Score = " + str(score))
    results.append([s, multiplier, trace_cluster_0, trace_cluster_1, det_cluster_0, det_cluster_1, cm, accuracy, recall, precision, score, score_svm, num_overlap])
    # Calculate F1-Score https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
     
    """   
    # Plot
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
    s, multiplier, trace_cluster_0, trace_cluster_1, det_cluster_0, det_cluster_1, cm, accuracy, recall, precision, score
#dump(results, '/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/experiment_result/experiment02_unbalanced_clusters.pkl')
results = pd.DataFrame(results,  columns=['Seed', 'Multiplier for STD','Trace Cluster 0', "Trace Cluster 1","Determinant Cluster 0", "Determinant Cluster 1","Confusion Matrix", 'Accuracy', 'Recall', 'Precision', 'Score', 'SVM F1-Score', 'Number of overlapping Dimensions'])
dump(results, '/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/experiment_result/experiment04_Spread_Differences__additional_Infos.pkl')