# import seaborn as sns
# import tensorflow as tf
import os
import numpy as np
import glob
from sklearn.decomposition import PCA
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, scale
from visualization.kmeans import kmeans, multipleKmeans

# from sklearn import model_selection
# from tensorboard.plugins import projector

from tqdm import tqdm
LENGTH = 40

def pca(input_directory):
    # To keep track of how many files we're processing
    count = 0

    videoIndexes = [0]
    totalDimensions = 0
    tags = []
    ids = []
    numbered_ids = []
    legendNames = []

    pbar = tqdm(total=LENGTH)

    for np_name in glob.glob(input_directory + '*.np[yz]'):
        if (count == 0):
            embeddings = np.load(np_name).T
            embeddings_shape = embeddings.shape[1]
        else:
            np_embedding = np.load(np_name).T
            embeddings_shape = np_embedding.shape[1]
            embeddings = np.concatenate([embeddings, np_embedding], axis=1)
        
        totalDimensions += embeddings_shape
        videoIndexes.append(totalDimensions)

        formatted_name = os.path.basename(np_name).removesuffix('.mp4.npy')
        legendNames.append(formatted_name)

        for i in range(embeddings_shape):
            tags.append(formatted_name + '_' +str(i))
            ids.append(formatted_name)
            numbered_ids.append(int(formatted_name))

        print("Processed: {}".format(np_name))

        count += 1

        pbar.update(n=1)
        if (count == LENGTH):
            break
    
    scaledEmbeddings = StandardScaler().fit_transform(embeddings)
    model = PCA(n_components=2)
    model_result = model.fit_transform(scaledEmbeddings)
    print('Explained variance per dimension:')
    for i, variance in enumerate(model.explained_variance_ratio_):
        print('Dimension {} accounts for {:.2%} in variance'.format(i, variance))
    print('Total explained variance: {:.2%}'.format(np.sum(model.explained_variance_ratio_)))
    print('PCA done')

    # multipleKmeans(scaledEmbeddings)
    kmeans(model_result, 3)
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
=======

    
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
>>>>>>> d4a10a4 (pca scatter plot added)
    print('Scatter figure complete')
    
    # Using seaborn to make a scatter plot
    """
    scatterplot = sns.scatterplot(
                    data=category_vector_frame, 
                    x='col1', 
                    y='col2', 
                    hue=tags, 
                    palette="deep")
    fig = scatterplot.get_figure()
    fig.savefig('result.png')
    """


    # Trying out pad_sequences
    """
    embeddings = tf.keras.preprocessing.sequence.pad_sequences(npy_files, padding='post')
    embedding_var = tf.Variable(embeddings, name='embedding_var')
    tf.reshape(embedding_var, [-1])
    print(embedding_var)"""
