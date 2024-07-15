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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from utils import (calculate_thresholds_gaussian_kernel,
                   generate_migrants_from_distribution, overlap_check)

"""
The Fritzens-Sanzeno/reduced dataset was published by Grupe, G., Klaut, D., Otto, L., Mauder, M., Lohrer, J., Kröger, P., and Lang, A. (2020).
The genesis and spread of the early Fritzens-Sanzeno culture (5th/4th cent. BCE)–Stable
isotope analysis of cremated and uncremated skeletal finds. Journal of Archaeological
Science: Reports, 29:102121. Publisher: Elsevier.

This should be cited accordingly when used
"""
# Read Isotope Data Set
# Set name of dataset
dataset_name = 'isotope_2-3'
isodata = pd.read_csv('Dataset.csv')
cols = ["87Sr/86Sr", "208Pb/204Pb", "207Pb/204Pb", "206Pb/204Pb", "208Pb/207Pb", "206Pb/207Pb"]

# For full data set
# Create Groups for North of Alps, Inneralpine, and southern Tirol
site_group = pd.DataFrame(isodata["site code"].values /100)
isodata['site group'] = site_group
isodata['site group'] = isodata['site group'].astype('int64')


# Select two groups
isodata = isodata.query('`site group` == 2 | `site group` == 3')
isodata.loc[isodata['site group'] == 2, 'site group'] =0
isodata.loc[isodata['site group'] == 3, 'site group'] =1

# Visualize Dataset
#sns.pairplot(isodata, vars=cols, hue="site group")

# Cross Validation and using different Kernels/Parameters
pipe = make_pipeline(MinMaxScaler(), SVC())
print(pipe.named_steps)

# Grid Search with 5-Fold Stratified Crossvalidation for Linear Kernel
kernel = ['linear']
C_range = np.logspace(-2, 10, 100)
param_grid = dict(svc__C = C_range, svc__kernel=kernel, svc__cache_size = [500])
grid_search_lin = GridSearchCV(pipe, param_grid=param_grid, scoring='f1', refit=True, cv=5, n_jobs=16, verbose=42)
grid_search_lin.fit(isodata[cols], isodata['site group'])
dump(grid_search_lin, "isotope_prediction/parameter_tuning/grid_search_" + dataset_name + "_linear_n100.pkl")


# Grid Search with 5-Fold Stratified Crossvalidation for rbf Kernel
kernel = ['rbf']
C_range = np.logspace(-2, 10, 100)
Gamma_range = np.logspace(-9, 4, 100)
param_grid = dict(svc__C=C_range, svc__gamma=Gamma_range, svc__kernel=kernel, svc__cache_size = [500])
grid_search_rbf = GridSearchCV(pipe, param_grid=param_grid, scoring='f1', refit=True, cv=5, n_jobs=16, verbose=42)
grid_search_rbf.fit(isodata[cols], isodata['site group'])
dump(grid_search_rbf, "isotope_prediction/parameter_tuning//grid_search_" + dataset_name + "_rbf_n100.pkl")


# Grid Search with 5-Fold Stratified Crossvalidation for Poly Kernel
# Max Iter for datasets 1-2 and 2-3 not used in other datasets
kernel = ['poly']
C_range = np.logspace(-2, 10, 20)
Gamma_range = np.logspace(-9, 4, 20)
param_grid = dict(svc__C=C_range, svc__gamma=Gamma_range, svc__degree=[2,3,4], svc__kernel=kernel, svc__cache_size = [500], svc__max_iter=[100000000])
grid_search_poly = GridSearchCV(pipe, param_grid=param_grid, scoring='f1', refit=True, cv=5, n_jobs=16, verbose=42)
grid_search_poly.fit(isodata[cols], isodata['site group'])
dump(grid_search_poly, "isotope_prediction/parameter_tuning//grid_search_" + dataset_name + "_poly_n20.pkl")


