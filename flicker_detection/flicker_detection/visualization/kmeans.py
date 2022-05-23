from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from visualization.graph import graph
# import pandas as pd

def multipleKmeans(dataSet):
    fullRange = range(1, 9)
    score = []
    for i in fullRange:
        differentKmeans = KMeans(n_clusters=i)
        score.append(differentKmeans.fit(dataSet).score(dataSet))
        print("Finished for {} iteration".format(i))
    print("Finished calculating K means for all")
    plt.plot(fullRange, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.savefig('./visualization/result.png')
    print("Plot done")
    return

def kmeans(PCA_result, k):
    labels = KMeans(n_clusters=k).fit_predict(PCA_result)
    unique_labels = np.unique(labels)
    
    graph(PCA_result, labels, unique_labels)
    return