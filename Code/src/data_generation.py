import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs


def generate_testdata(path):
    approx_std = [[0.0023975, 0.073625 , 0.015925 , 0.13005  , 0.002775 , 0.007925 ],
       [0.001375 , 0.07545  , 0.023525 , 0.081725 , 0.0023   , 0.005625 ]]
    
    mean = [[0.7207, 38.2853, 15.6458, 18.2447, 2.4545, 1.1645],
                          [0.70573, 39.0076, 15.8019, 18.9493, 2.4783, 1.2085]]

    X, clusters = make_blobs(n_samples=[150,150], centers=mean, cluster_std=approx_std, n_features=6, random_state=0)
    data = pd.DataFrame(X, columns=['87Sr/86Sr','208Pb/204Pb','207Pb/204Pb','206Pb/204Pb','208Pb/207Pb','206Pb/207Pb'])
    data['cluster'] = clusters
    data.to_pickle(path)


def generate_gaussian_dataset(n, means, stds, n_dims, rand=0):
     X, clusters = make_blobs(n_samples=n, centers=means, cluster_std=stds, n_features=n_dims, random_state=rand)
     data = pd.DataFrame(X, columns=['87Sr/86Sr','208Pb/204Pb','207Pb/204Pb','206Pb/204Pb','208Pb/207Pb','206Pb/207Pb'])
     data['cluster'] = clusters
     return data
