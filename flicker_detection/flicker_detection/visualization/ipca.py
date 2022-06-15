import numpy as np

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler

import plotly.graph_objects as go

from visualization.kmeans import kmeans, multipleKmeans
from visualization.mean_shift import meanshift
from visualization.data_manager import DataManager

def scatter_plot(data: DataManager, model_result):
    modelSplit = np.array_split(
                    model_result,
                    indices_or_sections=data.videoIndexes[1:-1],
                    axis=0)
    
    # Creates a figure widget
    f = go.FigureWidget()
    
    # Creates a trace per each video file embedding
    for splitCount, m in enumerate(modelSplit):
        mTranspose = np.transpose(m)
        scatterGl = go.Scattergl(
            name=data.legendNames[splitCount],
            x=mTranspose[0],
            y=mTranspose[1],
            mode='markers',
            ids=data.ids[data.videoIndexes[splitCount]:data.videoIndexes[splitCount+1]],
            hovertext=data.tags[data.videoIndexes[splitCount]:data.videoIndexes[splitCount+1]],
            hoverinfo='text',
            connectgaps=False
        )
        f.add_trace(scatterGl)
    
    f.write_html('./visualization/ipca.html')


def ipca(input_directory):
    KernelPcaData = DataManager(input_directory)
    scaledEmbeddings = MinMaxScaler().fit_transform(np.transpose(KernelPcaData.embeddings))
    # scaledEmbeddings = StandardScaler().fit_transform(np.transpose(KernelPcaData.embeddings))

    IPCA = IncrementalPCA(n_components=2, batch_size=400)
    model_result = IPCA.fit_transform(scaledEmbeddings)

    scatter_plot(KernelPcaData, model_result)

    return