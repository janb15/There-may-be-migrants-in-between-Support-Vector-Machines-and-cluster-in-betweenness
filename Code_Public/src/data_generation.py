import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd


def generate_testdata(path):
    #min = np.array([[0.70907, 38.314, 15.6147, 18.3392, 2.4513, 1.1733],
    #         [0.71012, 38.6657, 15.6452, 18.5191, 2.4682, 1.1784]])
    #max = np.array([[0.72266, 38.6085, 15.6784, 18.8594, 2.4664, 1.205],
    #        [0.71562, 38.9675, 15.7393, 18.846, 2.4774, 1.2009]])        
    #approx_std = (max-min)/4 # Range Rule: the std is approx equal to the range of the data divided by 4
    #mean = [[0.71537, 38.4853, 15.6478, 18.5347, 2.4595, 1.1845],
    #                       [0.71273, 38.8076, 15.6909, 18.6493, 2.4733, 1.1885]]
# Changes because of overlap
    approx_std = [[0.0023975, 0.073625 , 0.015925 , 0.13005  , 0.002775 , 0.007925 ],
       [0.001375 , 0.07545  , 0.023525 , 0.081725 , 0.0023   , 0.005625 ]]
    
    mean = [[0.7207, 38.2853, 15.6458, 18.2447, 2.4545, 1.1645],
                          [0.70573, 39.0076, 15.8019, 18.9493, 2.4783, 1.2085]]
       # Std nochmal reduzieren
       # Changes in 87Sr/86Sr Mean by +0.005 and -0.007 and Std by -0,001 and +-0
       # Changes in 208Pb/204Pb Mean by - 0.2 and + 0.2
       # Changes in 207Pb/204Pb Mean by -0.002 and +0.002
       # Changes in 206Pb/204Pb Mean by -0.3 and +0.3
       # Changes in 206Pb/207Pb Mean by - 0.2 and +0.2
       # Changes in 208Pb/207Pb Mean by - 0.005 and + 0.005std by cluster1 -0,001
    X, clusters = make_blobs(n_samples=[150,150], centers=mean, cluster_std=approx_std, n_features=6, random_state=0)
    data = pd.DataFrame(X, columns=['87Sr/86Sr','208Pb/204Pb','207Pb/204Pb','206Pb/204Pb','208Pb/207Pb','206Pb/207Pb'])
    data['cluster'] = clusters


    #group_start = cluster_0.iloc[np.random.choice(cluster_0.shape[0], 10, replace=False)]
    #group_end = cluster_1.iloc[np.random.choice(cluster_1.shape[0], 10, replace=False)]

    #group = [np.linspace(group_start.iloc[x], group_end.iloc[x],num=20) for x in np.arange(0,9)]
    data.to_pickle(path)
    #group # TODO: Genauer anschauen
    #data.head()     

def generate_gaussian_dataset(n, means, stds, n_dims, rand=0):
     X, clusters = make_blobs(n_samples=n, centers=means, cluster_std=stds, n_features=n_dims, random_state=rand)
     data = pd.DataFrame(X, columns=['87Sr/86Sr','208Pb/204Pb','207Pb/204Pb','206Pb/204Pb','208Pb/207Pb','206Pb/207Pb'])
     data['cluster'] = clusters
     return data


    #group_start = cluster_0.iloc[np.random.choice(cluster_0.shape[0], 10, replace=False)]
    #group_end = cluster_1.iloc[np.random.choice(cluster_1.shape[0], 10, replace=False)]

    #group = [np.linspace(group_start.iloc[x], group_end.iloc[x],num=20) for x in np.arange(0,9)]
     #data.to_pickle(path)
    #group # TODO: Genauer anschauen
    #data.head()     