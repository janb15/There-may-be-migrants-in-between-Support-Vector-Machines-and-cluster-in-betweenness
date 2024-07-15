# !!! USES  BURRIAL SITE LABEL FOR THRESHOLD CHECK!!!
# Train SVM on Isotope Set with Label = Site Group
# With Scaling before Split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
from utils import generate_migrants_from_distribution
from utils import calculate_thresholds_labelled
from utils import overlap_check
from train import train_lin_simple
from migrant_detection import predict_migrant_kernel_labelled
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

# Set the name of the experiment
experiment = "reduced_dataset_exhaustive_grid_best_2_per_kernel"

# Read the Isotope Data
isodata = pd.read_csv("path/Fritzens_Sanzeno_dataset.csv")
print(isodata.columns)
# Prepare result columns
isodata['Model'] = "None"
isodata['Earliest positive Prediction at Multiplier'] = -1
isodata['temporary prediction'] = 0
# Load the best models from gridsearch
#grid = load("/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/isotope_prediction/parameter_tuning/grid_search_isotope_reduced.pkl")
grid = load("/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/models/grid_search_isotope_reduced_combined_best2_per_kernel")
#models = pd.DataFrame(grid.cv_results_) # Used when not earlier converted to dataframe
models = grid.sort_values('mean_test_score', ascending=False)
best_models = models.head(10)
#sns.pairplot(isodata, vars=cols, hue="site group")
# params rbf c=1000 gamma=scale
# Possible Migrants - Grave Goods: Wörgl 71a, Wörgl 77 u.O., Wörgl 78, Kundl 89/I -
# Statistical Outliers: Kundl: 7, 25, 75, 89/I, 90, 139, 151; Wörgl: 72a, 77, 77 u.O. 78
# Suspicious Cluster Assignment: Kundl: 89/I, 89/II
# Third Cluster: Kundl: 160,40, 
# Overlap: Kundl 90, 42; Wörgl: 44, 72, 72a, 77 o.U.; Moritzing 5/T22, 16/T73, 11/T45, 30/T188, T3/18
# Burial Codes: [72a, 77 u.O., 78, 89/I, 89II, 7, 25, 75, 90, 139, 151, 72a, 77, 160, 40, 90, 42, 44, 72,  5/T22, 16/T73, 11/T45, 30/T188, T3/18]
# Read the Isotope Data

cols = ["87Sr/86Sr", "208Pb/204Pb", "207Pb/204Pb", "206Pb/204Pb", "208Pb/207Pb", "206Pb/207Pb"]

#sns.pairplot(isodata, vars=cols, hue="site group")
scaler = MinMaxScaler() #Can be changed to robustscaler or standardscaler
# Scale Data Set
isodata[cols] = scaler.fit_transform(isodata[cols])
#sns.pairplot(isodata, vars=cols, hue="site group")
reports = []
# Iterate through best models
for index, row in best_models.iterrows():
    print(row["param_svc__kernel"], row["param_svc__C"], row["param_svc__gamma"])
    if row["param_svc__kernel"] == 'linear':
         model = SVC(kernel=row["param_svc__kernel"], C = row["param_svc__C"], random_state=0, cache_size=4000)
    elif row["param_svc__kernel"] == 'rbf':
        model = SVC(kernel=row["param_svc__kernel"], C = row["param_svc__C"], gamma = row["param_svc__gamma"], random_state=0, cache_size=4000)
    else:
        model = SVC(kernel=row["param_svc__kernel"], C = row["param_svc__C"], gamma = row["param_svc__gamma"], degree = row["param_svc__degree"], random_state=0, cache_size=4000)
    model.fit(isodata[cols], isodata['site group'])
    dump(model, "/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/isotope_prediction/models/" + experiment + "Kernel: "+ str(row["param_svc__kernel"]) + " C: " + str(row["param_svc__C"]) + " Gamma: " + str(row["param_svc__gamma"]) + ".pkl")
    predictions = model.predict(isodata[cols]) # TODO: Visualisierung hier wahrscheinlich entfernen
    # Save classification report
    classif_report = classification_report(isodata["site group"], predictions)
    reports.append(classif_report)
    # Show confusion matrix
    #cm = confusion_matrix(isodata['site group'], predictions, labels=model.classes_)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    #disp.plot()
    #plt.show()

    # Rename burrial site column for threshold calculation
    isodata_cluster = isodata.copy()
    isodata_cluster = isodata_cluster.rename(columns={"site group": "cluster"})
    # Generate threshold for distance to hyperplane(t1) and distance to cluster connecting vector(t2)
    multipliers = np.linspace(3,0, num= 30).round(1)
    # Prepare results per model
    result_per_model = isodata.copy()
    result_per_model['Model'] = "Kernel: "+ str(row["param_svc__kernel"]) + " C: " + str(row["param_svc__C"]) + " Gamma: " + str(row["param_svc__gamma"])
    for mult in multipliers:
        t1_l, t2_l = calculate_thresholds_labelled(isodata_cluster[["87Sr/86Sr", "208Pb/204Pb", "207Pb/204Pb", "206Pb/204Pb", "208Pb/207Pb", "206Pb/207Pb", "cluster"]], model, mult)
        # Predict full data set
        data_predict_migrants = predict_migrant_kernel_labelled(isodata_cluster[cols].to_numpy(), isodata_cluster['cluster'].to_numpy(), isodata_cluster[cols], model, t1_l, t2_l, mult)
        n_known_migrants = isodata_cluster[isodata_cluster['Migrant']==1].shape[0]
        print("Multiplier: " + str(mult))
        print("Number of known Migrants: " + str(n_known_migrants))

        # Find Data to predicted migrants
        x = np.array(data_predict_migrants)
        indices = np.where(x == 1)[0]
        print("# Migrants predicted: " + str(len(indices)))
        print("# Samples in data set: " + str(isodata_cluster.shape[0]))
        #isodata.iloc[indices]
        isodata_cluster['prediction'] = x
        equal = isodata_cluster[isodata_cluster['prediction'] == isodata_cluster['Migrant']]
        unequal = isodata_cluster[isodata_cluster['prediction'] != isodata_cluster['Migrant']]
        print("Matching positive Predictions: " + str(equal[equal['prediction'] == 1]))
        print("# of matching predictions: " + str(equal[equal['prediction'] == 1].shape[0]))
        print("Not matching positive Predictions: " + str(unequal[unequal['prediction'] == 1]))
        print("# of notmatching positive predictions: " + str(unequal[unequal['prediction'] == 1].shape[0]))
        # Save results per model
        result_per_model['Multiplier ' +str(mult) + " prediction"] = x
        result_per_model['Multiplier ' +str(mult) + " num predicted migrants"] =  len(indices)
        result_per_model['Multiplier ' +str(mult) + " Matching positive Predictions"] =  equal[equal['prediction'] == 1].shape[0]
        result_per_model['Multiplier ' +str(mult) + " Not matching positive Predictions"] =  unequal[unequal['prediction'] == 1].shape[0]
        # Save earliest positive prediction per model
        result_per_model['temporary prediction'] = x
        result_per_model.loc[(result_per_model['temporary prediction'] == 1) & (result_per_model['Earliest positive Prediction at Multiplier'] < mult), 'Earliest positive Prediction at Multiplier'] = mult
        
        isodata['temporary prediction'] = x
        # Save results for all models
        # Save earliest positive prediction over all models
        isodata.loc[(isodata['temporary prediction'] == 1) & (isodata['Earliest positive Prediction at Multiplier'] < mult), 'Model'] = "Kernel: "+ str(row["param_svc__kernel"]) + " C: " + str(row["param_svc__C"]) + " Gamma: " + str(row["param_svc__gamma"])
        isodata.loc[(isodata['temporary prediction'] == 1) & (isodata['Earliest positive Prediction at Multiplier'] < mult), 'Earliest positive Prediction at Multiplier'] = mult
    dump(result_per_model, "/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/isotope_prediction/predictions/" + experiment + "Kernel: "+ str(row["param_svc__kernel"]) + " C: " + str(row["param_svc__C"]) + " Gamma: " + str(row["param_svc__gamma"]) + ".pkl") 

best_models['Classification Report'] = reports
dump(best_models, "/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/isotope_prediction/models/best_models_" + experiment + ".pkl")
dump(isodata, "/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/isotope_prediction/predictions/" + experiment + ".pkl")
