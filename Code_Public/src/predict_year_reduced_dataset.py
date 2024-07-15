from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
from joblib import dump, load
from utils import generate_migrants_from_distribution
from migrant_detection import predict_year
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt




# Load positive predictions and full dataset
migrant_candidates = load("/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/isotope_prediction/predictions/reduced_dataset_best10_positive_predictions.pkl")
isodata = pd.read_csv("/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/isotope_prediction/datasets/tabula-Grupe_cleaned.csv")

cols = ["87Sr/86Sr", "208Pb/204Pb", "207Pb/204Pb", "206Pb/204Pb", "208Pb/207Pb", "206Pb/207Pb"]

scaler = MinMaxScaler() #Can be changed to robustscaler or standardscaler
# Scale Data Set
isodata[cols] = scaler.fit_transform(isodata[cols])

sitegroup_0 = isodata[isodata['site group'] == 0]
sitegroup_1 = isodata[isodata['site group'] == 1]

mean_c0 = sitegroup_0[cols].mean(axis = 0)
mean_c1 = sitegroup_1[cols].mean(axis = 0)
migrant_candidates_reduced = migrant_candidates[["87Sr/86Sr", "208Pb/204Pb", "207Pb/204Pb", "206Pb/204Pb", "208Pb/207Pb", "206Pb/207Pb","site group"]]
year_predictions = [predict_year(mean_c0, mean_c1, x[:-1], x[-1]) for x in migrant_candidates_reduced.to_numpy()]
print(mean_c0)
print(mean_c1)
migrant_candidates['Predicted Years since Migration'] = year_predictions
print(migrant_candidates[["Site", "Burial" , "site group", 'Predicted Years since Migration']])
ax = sns.regplot(data=migrant_candidates, x=migrant_candidates.index, y="Predicted Years since Migration")
ax.set_xticks(range(0,len(migrant_candidates)), migrant_candidates["Burial"])
ax.set_xlabel("Burial")
plt.xticks(rotation=45, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Burial", fontsize=20)
plt.ylabel('Predicted Years since Migration', fontsize=20)

plt.show()