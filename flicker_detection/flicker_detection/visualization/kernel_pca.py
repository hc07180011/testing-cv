import numpy as np
from sklearn.decomposition import KernelPCA

from sklearn.preprocessing import MinMaxScaler
from visualization.kmeans import kmeans, multipleKmeans
from visualization.meanShift import meanshift
from visualization.data_manager import DataManager

def kernel_pca(input_directory):
    KernelPcaData = DataManager(input_directory)

    scaledEmbeddings = MinMaxScaler().fit_transform(KernelPcaData.embeddings)

    # Trying out multiple kernels to determine the best one
    kernel_list = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']

    # Also loop through both the list of kernels and different n_components sizes to determine the best one
    n_components_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    for kernel in kernel_list:
        for n_components in n_components_list:
            model = KernelPCA(n_components=n_components, kernel=kernel)
            model_result = model.fit_transform(scaledEmbeddings)
            print('Kernel PCA with {} and {} components done'.format(kernel, n_components))
            
            for i, variance in enumerate(model.explained_variance_ratio_):
                print('Dimension {} accounts for {:.2%} in variance'.format(i, variance))
            print('Total explained variance: {:.2%}'.format(np.sum(model.explained_variance_ratio_)))
            print()
    return