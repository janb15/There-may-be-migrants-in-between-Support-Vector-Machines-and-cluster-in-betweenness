import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
from migrant_detection import predict_year
from sklearn.datasets import make_blobs
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import MinMaxScaler
from utils import generate_migrants_from_distribution

"""
The Fritzens-Sanzeno/reduced dataset was published by Grupe, G., Klaut, D., Otto, L., Mauder, M., Lohrer, J., Kröger, P., and Lang, A. (2020).
The genesis and spread of the early Fritzens-Sanzeno culture (5th/4th cent. BCE)–Stable
isotope analysis of cremated and uncremated skeletal finds. Journal of Archaeological
Science: Reports, 29:102121. Publisher: Elsevier.

This should be cited accordingly when used
"""
# Load positive predictions and full dataset
migrant_candidates = load("isotope_prediction/predictions/reduced_dataset_best10_positive_predictions.pkl")
isodata = pd.read_csv("path/Fritzens-Sanzeno-Dataset.csv")

cols = ["87Sr/86Sr", "208Pb/204Pb", "207Pb/204Pb", "206Pb/204Pb", "208Pb/207Pb", "206Pb/207Pb"]

scaler = MinMaxScaler() #Can be changed to robustscaler or standardscaler
# Scale Data Set
isodata[cols] = scaler.fit_transform(isodata[cols])
# select site groups
sitegroup_0 = isodata[isodata['site group'] == 0]
sitegroup_1 = isodata[isodata['site group'] == 1]
# calculate means
mean_c0 = sitegroup_0[cols].mean(axis = 0)
mean_c1 = sitegroup_1[cols].mean(axis = 0)
migrant_candidates_reduced = migrant_candidates[["87Sr/86Sr", "208Pb/204Pb", "207Pb/204Pb", "206Pb/204Pb", "208Pb/207Pb", "206Pb/207Pb","site group"]]
# predict years
year_predictions = [predict_year(mean_c0, mean_c1, x[:-1], x[-1]) for x in migrant_candidates_reduced.to_numpy()]
migrant_candidates['Predicted Years since Migration'] = year_predictions
print(migrant_candidates[["Site", "Burial" , "site group", 'Predicted Years since Migration']])
# Plot results
ax = sns.regplot(data=migrant_candidates, x=migrant_candidates.index, y="Predicted Years since Migration")
ax.set_xticks(range(0,len(migrant_candidates)), migrant_candidates["Burial"])
ax.set_xlabel("Burial")
plt.xticks(rotation=45, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Burial", fontsize=20)
plt.ylabel('Predicted Years since Migration', fontsize=20)

plt.show()