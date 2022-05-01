import numpy as np
import json
import tensorflow as tf

# Read the data back out.


def decode_fn(record_bytes):
    jm = json.load(open('test.json', "r"))
    schema = dict(map(lambda x: (x, tf.io.FixedLenFeature(
        [], dtype=tf.string)), jm['data/TFRecords/0099.mp4']))
    return tf.io.parse_single_example(
        # Data
        record_bytes,
        schema
    )


if __name__ == "__main__":

    example_path = "0099.mp4.tfrecords"
    np.random.seed(0)
    jm = json.load(open('test.json', "r"))
    schema = dict(map(lambda x: (x, tf.io.FixedLenFeature(
        [], dtype=tf.float32)), jm['data/TFRecords/0099.mp4']))
    ds = tf.data.TFRecordDataset(example_path)

    # for batch in ds.map(decode_fn):
    #     print(tf.io.parse_tensor(batch, out_type=tf.double))
    data = next(ds.as_numpy_iterator())
    parsed = tf.io.parse_tensor(data, out_type=tf.float32)
    # print(parsed)
