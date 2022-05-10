# import seaborn as sns
# import tensorflow as tf
import os
import numpy as np
import glob
from sklearn.decomposition import PCA
import pandas as pd
import plotly.graph_objects as go

# from sklearn import model_selection
# from tensorboard.plugins import projector

from tqdm import tqdm
LENGTH = 181

def pca(input_directory):
    # To keep track of how many files we're processing
    count = 0

    videoIndexes = []
    totalDimensions = 0
    tags = []
    ids = []
    numbered_ids = []

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

        for i in range(embeddings_shape):
            tags.append(formatted_name + '_' +str(i))
            ids.append(formatted_name)
            numbered_ids.append(int(formatted_name))

        print("Processed: {}".format(np_name))
        print("Size: {}".format(embeddings_shape))

        count += 1

        pbar.update(n=1)

    
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
