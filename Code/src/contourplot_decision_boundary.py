import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
from matplotlib.colors import LinearSegmentedColormap
from migrant_detection import predict_migrant_linear_dynamic_threshold
from utils import calculate_thresholds_gaussianl

dimensions = ['87Sr/86Sr','208Pb/204Pb','207Pb/204Pb','206Pb/204Pb','208Pb/207Pb','206Pb/207Pb']
# Fine granularity uses a lot of memory, GRANULARITY_FIRST² * GRANULARITY_SECOND⁴ is the size of the arrays.
GRANULARITY = 15
# Load SVM Model - Insert Path
model = load('path/model.pkl')
# Load labeled Dataset
data_labelled = load("path/dataset.pkl")
data = data_labelled.drop(labels='cluster',axis= 'columns')
# Calculate Thresholds
t1, t2 = calculate_thresholds_gaussian(data_labelled, model, 3)
# Generate Grid
boundaries = list(zip(list(data.min()),list(data.max())))
x1 = np.linspace(boundaries[0][0], boundaries[0][1], GRANULARITY)
x2 = np.linspace(boundaries[1][0], boundaries[1][1], GRANULARITY)
x3 = np.linspace(boundaries[2][0], boundaries[2][1], GRANULARITY)
x4 = np.linspace(boundaries[3][0], boundaries[3][1], GRANULARITY)
x5 = np.linspace(boundaries[4][0], boundaries[4][1], GRANULARITY)
x6 = np.linspace(boundaries[5][0], boundaries[5][1], GRANULARITY)
X1, X2 , X3, X4, X5, X6= np.meshgrid(x1,x2,x3,x4,x5,x6)
xy = np.vstack([X1.ravel(), X2.ravel(), X3.ravel(), X4.ravel(), X5.ravel(), X6.ravel()]).T  

# Select which kind of Contour should be generated and calculate values for grid samples
#P = model.decision_function(xy)
#P = predict_migrant_linear(xy, data.copy(), model, t1, t2)
P = predict_migrant_linear_dynamic_threshold(xy, data.copy(), model, t1, t2)

# Plot
df = pd.DataFrame(xy, columns=['87Sr/86Sr','208Pb/204Pb','207Pb/204Pb','206Pb/204Pb','208Pb/207Pb','206Pb/207Pb'])
support = np.append(np.array(model.support_vectors_),np.full((len(model.support_vectors_),1),3),axis=1)
#data = data.drop(columns=['Prediction'])
color = model.predict(data)
data['color'] = color
data = pd.concat([data,pd.DataFrame(support, columns=['87Sr/86Sr','208Pb/204Pb','207Pb/204Pb','206Pb/204Pb','208Pb/207Pb','206Pb/207Pb', 'color'])])
df['val'] = P
dim_combinations = itertools.permutations(dimensions,2)
fig, axs = plt.subplots(len(dimensions)-1,len(dimensions)-1, layout="constrained")
colors = [(0,0,255), (139,0,0), (255,255,0)]
cmap = LinearSegmentedColormap.from_list('own_map', colors, N=3)
for ax in axs.flat:
   dims = next(dim_combinations)
   grouped = df.groupby(list(dims),sort=False, as_index=False).mean()
   cs = ax.contourf(pd.unique(grouped[dims[0]]), pd.unique(grouped[dims[1]]), np.array(grouped['val']).reshape((GRANULARITY,GRANULARITY)))#, cmap='Paired')
   fig.colorbar(cs,ax=ax,shrink=0.9)
   #ax.scatter(data[dims[0]],data[dims[1]],c=data['color'], cmap=cmap)
   sns.scatterplot(data_labelled, x=dims[0], y=dims[1], hue='cluster',ax=ax, palette='colorblind')
   ax.get_legend().set_visible(False)
   #ax.set(xlabel=dims[0],ylabel=dims[1])
plt.show()
