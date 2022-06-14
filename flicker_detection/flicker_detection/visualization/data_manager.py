import os
import numpy as np
import glob

from tqdm import tqdm

LENGTH = 181

class DataManager:
    def __init__(self, input_directory):
        self.input_directory = input_directory
        self.videoIndexes = [0]
        self.totalDimensions = 0
        self.tags = []
        self.ids = []
        self.numbered_ids = []
        self.legendNames = []

        pbar = tqdm(total=LENGTH)

        for count, np_name in enumerate(glob.glob(self.input_directory + '*.np[yz]')):
            if (count == 0):
                self.embeddings = np.load(np_name).T
                self.embeddings_shape = self.embeddings.shape[1]
            else:
                np_embedding = np.load(np_name).T
                self.embeddings_shape = np_embedding.shape[1]
                self.embeddings = np.concatenate([self.embeddings, np_embedding], axis=1)
            
            self.totalDimensions += self.embeddings_shape
            self.videoIndexes.append(self.totalDimensions)

            formatted_name = os.path.basename(np_name).removesuffix('.mp4.npy')
            self.legendNames.append(formatted_name)

            for i in range(self.embeddings_shape):
                self.tags.append(formatted_name + '_' +str(i))
                self.ids.append(formatted_name)
                self.numbered_ids.append(int(formatted_name))

            pbar.update(n=1)
            if (count == LENGTH):
                break