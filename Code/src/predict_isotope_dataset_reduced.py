import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
from migrant_detection import predict_migrant_kernel_labelled
from sklearn.datasets import make_classification
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from train import train_lin_simple
from utils import (calculate_thresholds_labelled,
                   generate_migrants_from_distribution, overlap_check)

"""
The Fritzens-Sanzeno/reduced dataset was published by Grupe, G., Klaut, D., Otto, L., Mauder, M., Lohrer, J., Kröger, P., and Lang, A. (2020).
The genesis and spread of the early Fritzens-Sanzeno culture (5th/4th cent. BCE)–Stable
isotope analysis of cremated and uncremated skeletal finds. Journal of Archaeological
Science: Reports, 29:102121. Publisher: Elsevier.

This should be cited accordingly when used
"""
# Set the name of the experiment
experiment = "reduced_dataset_exhaustive_grid_best_2_per_kernel"

# Read the Isotope Data
isodata = pd.read_csv("path/Fritzens_Sanzeno_dataset.csv")
print(isodata.columns)
# Prepare result columns
isodata['Model'] = "None"
isodata['Earliest positive Prediction at Multiplier'] = -1
isodata['temporary prediction'] = 0
# Load the best models from gridsearch - combined gridsearch results need to be generated with script in testing.ipynb
# Adapt grid path if needed
grid = load("/isotope_prediction/parameter_tuning/grid_search_isotope_reduced.pkl")
# models = pd.DataFrame(grid.cv_results_) # Used when not earlier converted to dataframe
models = grid.sort_values('mean_test_score', ascending=False)
best_models = models.head(10)


cols = ["87Sr/86Sr", "208Pb/204Pb", "207Pb/204Pb", "206Pb/204Pb", "208Pb/207Pb", "206Pb/207Pb"]

scaler = MinMaxScaler() #Can be changed to robustscaler or standardscaler
# Scale Data Set
isodata[cols] = scaler.fit_transform(isodata[cols])
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
    dump(model, "isotope_prediction/models/" + experiment + "Kernel: "+ str(row["param_svc__kernel"]) + " C: " + str(row["param_svc__C"]) + " Gamma: " + str(row["param_svc__gamma"]) + ".pkl")
    predictions = model.predict(isodata[cols])
    # Save classification report
    classif_report = classification_report(isodata["site group"], predictions)
    reports.append(classif_report)
    # Show confusion matrix
    #cm = confusion_matrix(isodata['site group'], predictions, labels=model.classes_)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    #disp.plot()
    #plt.show()

    # Rename burial site column for threshold calculation
    isodata_cluster = isodata.copy()
    isodata_cluster = isodata_cluster.rename(columns={"site group": "cluster"})
    # Generate threshold for distance to hyperplane(t1) and distance to cluster connecting vector(t2)
    multipliers = np.linspace(3,0, num= 30).round(1)
    # Prepare results per model
    result_per_model = isodata.copy()
    result_per_model['Model'] = "Kernel: "+ str(row["param_svc__kernel"]) + " C: " + str(row["param_svc__C"]) + " Gamma: " + str(row["param_svc__gamma"])
    # Predict for different multiplier values
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
    dump(result_per_model, "isotope_prediction/predictions/" + experiment + "Kernel: "+ str(row["param_svc__kernel"]) + " C: " + str(row["param_svc__C"]) + " Gamma: " + str(row["param_svc__gamma"]) + ".pkl") 

best_models['Classification Report'] = reports
dump(best_models, "isotope_prediction/models/best_models_" + experiment + ".pkl")
dump(isodata, "isotope_prediction/predictions/" + experiment + ".pkl")
