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
from utils import calculate_thresholds_gaussian_kernel
from utils import overlap_check
from train import train_lin_simple
from migrant_detection import predict_migrant_kernel_labelled
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Read Isotope Data Set
#isodata = pd.read_csv("/home/jan/CAU/Masterarbeit/Datenbanktabelle_cleaned.csv")
#dataset_name = 'isotope_reduced'
dataset_name = 'isotope_2-3'
#isodata = pd.read_csv("/home/jan/CAU/Masterarbeit/tabula-Grupe_cleaned.csv")
isodata = pd.read_csv('/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/isotope_prediction/datasets/Datenbanktabelle_cleaned.csv')
cols = ["87Sr/86Sr", "208Pb/204Pb", "207Pb/204Pb", "206Pb/204Pb", "208Pb/207Pb", "206Pb/207Pb"]

# For full data set
# Create Groups for North of Alps, Inneralpine, and southern Tirol
site_group = pd.DataFrame(isodata["site code"].values /100)
isodata['site group'] = site_group
isodata['site group'] = isodata['site group'].astype('int64')
# Delete single outlier
#isodata[isodata['206Pb/207Pb']>1.3]
#isodata.drop(isodata[isodata['206Pb/207Pb']>1.3].index, inplace=True)
#isodata[isodata['206Pb/207Pb']>1.3]


# Select two groups
isodata = isodata.query('`site group` == 2 | `site group` == 3')
isodata.loc[isodata['site group'] == 2, 'site group'] =0
isodata.loc[isodata['site group'] == 3, 'site group'] =1


#sns.pairplot(isodata, vars=cols, hue="site group")

# Cross Validation and using different Kernels/Parameters
# 24 min training time
pipe = make_pipeline(MinMaxScaler(), SVC())
print(pipe.named_steps)
#linear

kernel = ['linear']
C_range = np.logspace(-2, 10, 100)
param_grid = dict(svc__C = C_range, svc__kernel=kernel, svc__cache_size = [500])
grid_search_lin = GridSearchCV(pipe, param_grid=param_grid, scoring='f1', refit=True, cv=5, n_jobs=16, verbose=42)
grid_search_lin.fit(isodata[cols], isodata['site group'])
dump(grid_search_lin, "/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/models/grid_search_" + dataset_name + "_linear_n100.pkl")


# rbf
kernel = ['rbf']
C_range = np.logspace(-2, 10, 100)
Gamma_range = np.logspace(-9, 4, 100)
param_grid = dict(svc__C=C_range, svc__gamma=Gamma_range, svc__kernel=kernel, svc__cache_size = [500])
grid_search_rbf = GridSearchCV(pipe, param_grid=param_grid, scoring='f1', refit=True, cv=5, n_jobs=16, verbose=42)
grid_search_rbf.fit(isodata[cols], isodata['site group'])
dump(grid_search_rbf, "/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/models/grid_search_" + dataset_name + "_rbf_n100.pkl")

# Max Iter for 1-2 and 2-3
kernel = ['poly']
C_range = np.logspace(-2, 10, 20)
Gamma_range = np.logspace(-9, 4, 20)
param_grid = dict(svc__C=C_range, svc__gamma=Gamma_range, svc__degree=[2,3,4], svc__kernel=kernel, svc__cache_size = [500], svc__max_iter=[100000000])
grid_search_poly = GridSearchCV(pipe, param_grid=param_grid, scoring='f1', refit=True, cv=5, n_jobs=16, verbose=42)
grid_search_poly.fit(isodata[cols], isodata['site group'])
dump(grid_search_poly, "/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/models/grid_search_" + dataset_name + "_poly_n20.pkl")

#np.append(['auto','scale'],Gamma_range).astype(object)
# For Poly
#param_grid = dict(svc__kernel= kernel ,svc__C=[0.1, 0.3, 0.5, 0.7, 0.8, 1, 3, 5, 8, 10, 25, 100, 1000,10E10], svc__degree = [2,3,4],
#                   svc__gamma=['auto','scale', ], svc__cache_size = [500])

# For rbf and linear
#param_grid = dict(svc__kernel= kernel ,svc__C=[0.1, 0.3, 0.5, 0.7, 0.8, 1, 3, 5, 8, 10, 25, 100, 1000,10E10], 
#                   svc__gamma=[0.0001, 0.001, 0.1, 0.3, 0.5, 0.7, 0.8, 1, 3, 5, 8, 10, 25, 100, 1000,10E10, 'auto','scale', ], svc__cache_size = [500], )

#grid_search_lin.fit(isodata[cols], isodata['site group'])



"""
dump(grid_search, "/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/models/grid_search_isotope_reduced_rbf_n500.pkl")
print(
    "The best parameters are %s with a score of %0.2f"
    % (grid_search.best_params_, grid_search.best_score_)
)
predictions = grid_search.predict(isodata[cols])
print(classification_report(isodata['site group'],predictions))
cm = confusion_matrix(isodata['site group'], predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#grid_search.cv_results_


"""