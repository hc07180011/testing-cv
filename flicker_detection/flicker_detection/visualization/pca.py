# import seaborn as sns
# import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, scale
from visualization.kmeans import kmeans, multipleKmeans
from visualization.mean_shift import meanshift
from visualization.data_manager import DataManager

def pca(input_directory):
    PcaData = DataManager(input_directory)
    
    scaledEmbeddings = StandardScaler().fit_transform(PcaData.embeddings)
    model = PCA(n_components=2)
    model_result = model.fit_transform(scaledEmbeddings)
    print('Explained variance per dimension:')
    for i, variance in enumerate(model.explained_variance_ratio_):
        print('Dimension {} accounts for {:.2%} in variance'.format(i, variance))
    print('Total explained variance: {:.2%}'.format(np.sum(model.explained_variance_ratio_)))
    print('PCA done')

    meanshift(model_result, scaledEmbeddings)
    return

    # Splits into the original arrays
    
    modelSplit = np.array_split(
                    model.components_, 
                    indices_or_sections=videoIndexes[1:-1],
                    axis=1)
    
    # Creates a figure widget
    f = go.FigureWidget()
    f.layout.hovermode = 'closest'
    f.layout.hoverdistance = -1
    # Creates a trace per each video file embedding
    splitCount = 1
    for m in modelSplit:
        scatterGl = go.Scattergl(
            name=legendNames[splitCount-1],
            x=m[0],
            y=m[1],
            mode='markers',
            ids=ids[videoIndexes[splitCount-1]:videoIndexes[splitCount]],
            hovertext=tags[videoIndexes[splitCount-1]:videoIndexes[splitCount]],
            hoverinfo='text',
            connectgaps=False
        )
        f.add_trace(scatterGl)
        splitCount += 1

    def update_trace(trace, points, selector):
        if len(points.point_inds)==1:
            i = points.trace_index
            for x in range(0,len(f.data)):
                f.data[x]['marker']['color'] = 'grey'
                f.data[x]['opacity'] = 0.3
            #print('Correct Index: {}',format(i))
            f.data[i]['marker']['color'] = 'red'
            f.data[i]['opacity'] = 1

    for x in range(0,len(f.data)):
        f.data[x].on_click(update_trace)

    # Creates a dataframe of all components
    """
    category_vectors = model.components_.T
    category_vector_frame=pd.DataFrame(category_vectors, 
                                   columns=['col1', 'col2']).reset_index()
    """
    
    # Using Plotly to make a scatter plot
    # scatterGl = go.Scattergl(
    #     x=category_vector_frame['col1'],
    #     y=category_vector_frame['col2'],
    #     mode='markers',
    #     marker_color=numbered_ids,
    #     marker=dict(
    #         colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']]
    #     ),
    #     ids=ids,
    #     hovertext=tags,
    #     hoverinfo='text'
    # )

    # fig = go.Figure(scatterGl)
    # fig.write_html('./visualization/results.html')

    f.write_html('./visualization/results.html')

    
    model = PCA(n_components=2).fit(embeddings)
    print('PCA done')

    # Splits into the original arrays
    """
    modelSplit = np.array_split(
                    model.components_, 
                    indices_or_sections=videoIndexes[:-1],
                    axis=1)
    """

    category_vectors = model.components_.T
    category_vector_frame=pd.DataFrame(category_vectors, 
                                   columns=['col1', 'col2']).reset_index()
    
    # Using Plotly to make a scatter plot
    scatterGl = go.Scattergl(
        x=category_vector_frame['col1'],
        y=category_vector_frame['col2'],
        mode='markers',
        marker_color=numbered_ids,
        ids=ids,
        hovertext=tags,
        hoverinfo='text'
    )
    fig = go.Figure(scatterGl)
    fig.write_html('./visualization/results.html')

    print('Scatter figure complete')