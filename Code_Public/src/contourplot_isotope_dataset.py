import matplotlib.pyplot as plt
import numpy as np
from migrant_detection import predict_migrant_kernel_labelled
from utils import calculate_thresholds_labelled
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import itertools
from joblib import load, dump

import seaborn as sns

dimensions = ['87Sr/86Sr','208Pb/204Pb','207Pb/204Pb','206Pb/204Pb','208Pb/207Pb','206Pb/207Pb']
# Fine granularity uses a lot of memory, GRANULARITY_FIRST² * GRANULARITY_SECOND⁴ is the size of the arrays.
# TODO: Refactor for better memory usage
GRANULARITY = 12
model = load("/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/isotope_prediction/models/reduced_dataset_exhaustive_gridKernel: rbf C: 265608778.2946684 Gamma: 3.5111917342151347e-06.pkl")
#model = load("/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/isotope_prediction/models/reduced_dataset_exhaustive_grid_best_2_per_kernelKernel: poly C: 0.7847599703514611 Gamma: 3.792690190732254.pkl")
#model = load("/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/isotope_prediction/models/reduced_dataset_exhaustive_grid_best_2_per_kernelKernel: linear C: 705.4802310718645 Gamma: nan.pkl")
data = pd.read_csv('/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/isotope_prediction/datasets/tabula-Grupe_cleaned.csv')
scaler = MinMaxScaler() #Can be changed to robustscaler or standardscaler
# Scale Data Set
data[dimensions] = scaler.fit_transform(data[dimensions])
# Generate basic threshold for distance to hyperplane(t1) and distance to cluster connecting vector(t2)
 # Rename burrial site column for threshold calculation
isodata_cluster = data.copy()
isodata_cluster = isodata_cluster.rename(columns={"site group": "cluster"})
#t1, t2 = calculate_thresholds_basic(data_labelled, model) #basic thresholds
#t2 = np.max(t2)
mult= 1
t1_l, t2_l = calculate_thresholds_labelled(isodata_cluster[["87Sr/86Sr", "208Pb/204Pb", "207Pb/204Pb", "206Pb/204Pb", "208Pb/207Pb", "206Pb/207Pb", "cluster"]], model, mult)
#data_predict_migrants = predict_migrant_kernel_labelled(isodata_cluster[dimensions].to_numpy(), isodata_cluster['cluster'].to_numpy(), isodata_cluster[dimensions], model, t1_l, t2_l, mult)
data = data[dimensions]
boundaries = np.array(list(zip(list(data[dimensions].min()),list(data[dimensions].max()))))#[[data, 0.725], [38, 39,5], [15.5, 15,8], [18, 19], [2.44, 2.5], [1.1, 1.25] ]
# for scaled datasets the boundaries are 0 and 1. We add a little padding
boundaries = boundaries + [-0.1,0.1]
boundaries1 = [data[dimensions].min(axis=1)-data[dimensions].std(axis=1), data[dimensions].max(axis=1)+data[dimensions].std(axis=1)]
#plot_svc_decision_functin(model, boundaries)
x1 = np.linspace(boundaries[0][0], boundaries[0][1], GRANULARITY)
x2 = np.linspace(boundaries[1][0], boundaries[1][1], GRANULARITY)
x3 = np.linspace(boundaries[2][0], boundaries[2][1], GRANULARITY)
x4 = np.linspace(boundaries[3][0], boundaries[3][1], GRANULARITY)
x5 = np.linspace(boundaries[4][0], boundaries[4][1], GRANULARITY)
x6 = np.linspace(boundaries[5][0], boundaries[5][1], GRANULARITY)
X1, X2 , X3, X4, X5, X6= np.meshgrid(x1,x2,x3,x4,x5,x6)
xy = np.vstack([X1.ravel(), X2.ravel(), X3.ravel(), X4.ravel(), X5.ravel(), X6.ravel()]).T  
shap = xy.shape

# Visualize the decision boundary
#P_decision_function = model.decision_function(xy)
# Every threshold is calculated with the label equal to the prediction - visualize the migrant space
xy_label = model.predict(xy)
P_ibi_space = predict_migrant_kernel_labelled(xy, xy_label, isodata_cluster[dimensions], model, t1_l, t2_l, mult)
# Every threshold is calculated as if that point has burial = 0
#P = predict_migrant_kernel_labelled(xy, np.zeros(shape=(xy.shape[0],1)), isodata_cluster[dimensions], model, t1_l, t2_l, mult)
#P = linear_kernel_distance(xy, model)#.reshape((GRANULARITY_FIRST,GRANULARITY_FIRST, GRANULARITY_SECOND**4))
#P = predict_migrant_linear(xy, data.copy(), model, t1, t2)
#P = predict_migrant_linear_dynamic_threshold(xy, data.copy(), model, t1, t2)

df = pd.DataFrame(xy, columns=['87Sr/86Sr','208Pb/204Pb','207Pb/204Pb','206Pb/204Pb','208Pb/207Pb','206Pb/207Pb'])
#support = np.append(np.array(model.support_vectors_),np.full((len(model.support_vectors_),1),3),axis=1) #uncomment for support vectors
#data = data.drop(columns=['Prediction'])

#color = model.predict(data) # For color based on prediction
color = isodata_cluster['cluster'] # for color based on site group

data['color'] = color
#uncomment following line for support vector visualization
#data = pd.concat([data,pd.DataFrame(support, columns=['87Sr/86Sr','208Pb/204Pb','207Pb/204Pb','206Pb/204Pb','208Pb/207Pb','206Pb/207Pb', 'color'])])

"""dim_combinations = itertools.permutations(dimensions,2)
fig, axs = plt.subplots(len(dimensions)-1,len(dimensions)-1, layout="constrained")
colors = [(0,0,255), (139,0,0), (255,255,0)]
cmap = LinearSegmentedColormap.from_list('own_map', colors, N=3)
for ax in axs.flat:
   dims = next(dim_combinations)
   grouped = df.groupby(list(dims),sort=False, as_index=False).mean()
   cs = ax.contourf(pd.unique(grouped[dims[0]]), pd.unique(grouped[dims[1]]), np.array(grouped['val']).reshape((GRANULARITY,GRANULARITY)), cmap ='Reds')#, cmap='Paired')
   fig.colorbar(cs,ax=ax,shrink=0.9)
   ax.scatter(data[dims[0]],data[dims[1]],c=data['color'], cmap=cmap)
   ax.set(xlabel=dims[0],ylabel=dims[1])
#plt.savefig("contourplot_normalized_simple_lin_decision_function.svg" # speicher grafik zu klein
plt.show()
"""
"""
fig, ax = plt.subplots(layout='constrained')
colors = [(0,0,255), (139,0,0), (255,255,0)]
cmap = LinearSegmentedColormap.from_list('own_map', colors, N=3)
dim1 = '87Sr/86Sr'
dim2 = '208Pb/204Pb'

site_color = pd.factorize(isodata_cluster['Site'])[0]
grouped = df.groupby([dim1,dim2],sort=False, as_index=False).mean()
cs = plt.contourf(pd.unique(grouped[dim1]), pd.unique(grouped[dim2]), np.array(grouped['val']).reshape((GRANULARITY,GRANULARITY)), cmap ='Reds')
ax.scatter(data[dim1],data[dim2],c=data['color'], marker=site_color)
ax.set(xlabel=dim1,ylabel=dim2)
plt.show()
"""
# Import Fritzens-Sanzeno Candidates for labeling
fritzens_sanzeno_candidates = load('/home/jan/CAU/Masterarbeit/master-thesis-jan-bensien/Code/isotope_prediction/datasets/Fritzens_Sanzeno_Candidates.pkl')

# sns try

dim1 = '87Sr/86Sr'
dim2 = '208Pb/204Pb'
#data['site_color'] = pd.factorize(isodata_cluster['Site'])[0]
data['Region'] = isodata_cluster['cluster']
data['Burial Site'] =isodata_cluster['Site']
replacement_map = {
    1: "Inner Alpine",
    0: "South Tyrolean"
}
"""
fig, axs = plt.subplots(1,2,layout='constrained')
# For combined Figures
df['val'] = P_decision_function
data['Region'] = data['Region'].map(replacement_map)
grouped = df.groupby([dim1,dim2],sort=False, as_index=False).mean()
cs = axs[0].contourf(pd.unique(grouped[dim1]), pd.unique(grouped[dim2]), np.array(grouped['val']).reshape((GRANULARITY,GRANULARITY)))#, cmap ='Greens')
fig.colorbar(cs,ax=axs[0],shrink=0.9)
ax = sns.scatterplot(data=data, x=dim1, y=dim2, hue ='Region', s=150, style='Burial Site', ax=axs[0], palette='colorblind')
# Add the labels to the Fritzens-Sanzeno Candidates
for index, row in fritzens_sanzeno_candidates.iterrows():
    ax.text(row[dim1]+0.01, row[dim2], row['Burial'], horizontalalignment='left', size='medium', color='black')

df['val'] = P_ibi_space
grouped = df.groupby([dim1,dim2],sort=False, as_index=False).mean()
cs = axs[1].contourf(pd.unique(grouped[dim1]), pd.unique(grouped[dim2]), np.array(grouped['val']).reshape((GRANULARITY,GRANULARITY)))#, cmap ='Greens')
fig.colorbar(cs,ax=axs[1],shrink=0.9)
ax = sns.scatterplot(data=data, x=dim1, y=dim2, hue ='Region', s=150, style='Burial Site', ax=axs[1], palette='colorblind')
# Add the labels to the Fritzens-Sanzeno Candidates
for index, row in fritzens_sanzeno_candidates.iterrows():
    ax.text(row[dim1]+0.01, row[dim2], row['Burial'], horizontalalignment='left', size='medium', color='black')
#ax.set(xlabel=dim1,ylabel=dim2)
plt.show()
"""
fig, axs = plt.subplots(layout='constrained')
# For separated Figures
df['val'] = P_ibi_space
data['Region'] = data['Region'].map(replacement_map)
grouped = df.groupby([dim1,dim2],sort=False, as_index=False).mean()
cs = axs.contourf(pd.unique(grouped[dim1]), pd.unique(grouped[dim2]), np.array(grouped['val']).reshape((GRANULARITY,GRANULARITY)))#, cmap ='Greens')
fig.colorbar(cs,ax=axs,shrink=0.9)
ax = sns.scatterplot(data=data, x=dim1, y=dim2, hue ='Region', s=150, style='Burial Site', ax=axs, palette='colorblind')
# Add the labels to the Fritzens-Sanzeno Candidates
for index, row in fritzens_sanzeno_candidates.iterrows():
    ax.text(row[dim1]+0.01, row[dim2], row['Burial'], horizontalalignment='left', size='medium', color='black')
plt.legend(fontsize='x-large', title_fontsize='x-large')
plt.xlabel(xlabel=dim1, fontsize='xx-large')
plt.ylabel(ylabel=dim2, fontsize='xx-large')
plt.show()