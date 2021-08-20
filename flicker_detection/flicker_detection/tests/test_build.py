import os
import sys
import inspect
import numpy as np

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
os.chdir(os.path.join(os.getcwd(), ".."))


def test_alive():
    return True


def test_preprocess():
    from preprocessing.feature_extraction import Features
    video_features = Features(os.path.join(
        "tests", "test_data.mp4"), False, ".cache")
    video_features.feature_extraction()
    __cache = np.load(os.path.join(
        "tests", "52c4b8563b48b6025cd2d368eab2e33e.npz"))
    embeddings, suspects, horizontal_displacements, vertical_displacements = [
        __cache[__cache.files[i]] for i in range(len(__cache.files))]
    assert np.all(embeddings.shape == video_features.embeddings.shape)
    assert np.all(suspects == video_features.suspects)
    assert np.all(horizontal_displacements ==
                  video_features.horizontal_displacements)
    assert np.all(vertical_displacements ==
                  video_features.vertical_displacements)
