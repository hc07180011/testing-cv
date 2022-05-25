from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
from visualization.graph import graph

def meanshift(model_result, dataSet):
    bandwidth = estimate_bandwidth(dataSet, quantile=0.3, n_samples=800)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(dataSet)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    unique_labels = np.unique(labels)
    print("{} unique clusters found".format(len(unique_labels)))
    graph(model_result, labels, unique_labels)
    return