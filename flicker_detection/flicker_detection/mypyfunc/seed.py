import os
import torch
import random as rn
import numpy as np

DEFAULT_RANDOM_SEED = 2021


def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# tensorflow random seed
def seedTF(seed=DEFAULT_RANDOM_SEED):
    tf.random.set_seed(seed)


# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# basic + tensorflow + torch


def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTF(seed)
    seedTorch(seed)


def reset_random_seeds(seed=12345):
    """
    https://stackoverflow.com/questions/60058588/tesnorflow-2-0-tf-random-set-seed-not-working-since-i-am-getting-different-resul
    """
    import os
    # *IMPORANT*: Have to do this line *before* importing tensorflow
    os.environ['PYTHONHASHSEED'] = str(seed)

    import random as rn
    import numpy as np
    import os
    import tensorflow as tf
    from tensorflow.compat.v1.keras import backend as K

    tf.random.set_seed(seed)
    np.random.seed(seed)
    rn.seed(seed)

    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=gpu_options)

    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(), config=session_conf)

    K.set_session(sess)


if __name__ == "__main__":
    seedEverything()
    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

    import tensorflow as tf
    from tensorflow.compat.v1.keras import backend as K
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    np.random.seed(DEFAULT_RANDOM_SEED)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    rn.seed(DEFAULT_RANDOM_SEED)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

    tf.random.set_seed(DEFAULT_RANDOM_SEED)

    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(), config=session_conf)
    K.set_session(sess)
