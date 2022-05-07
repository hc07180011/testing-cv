import numpy as np
import tensorflow as tf

# Read the data back out.


def decode_fn(record_bytes, key):
    string = tf.io.parse_single_example(
        record_bytes,
        {key: tf.io.FixedLenFeature([], dtype=tf.string), }
    )
    return tf.io.parse_tensor(string[key], out_type=tf.float32)


if __name__ == "__main__":

    example_path = "TFRecords/0099.mp4.tfrecords"
    np.random.seed(0)
    ds = tf.data.TFRecordDataset(example_path).map(
        lambda byte: decode_fn(byte, "data/TFRecords/0099.mp4"))
    record = ds.get_single_element()
    # tensor = tf.io.parse_tensor(
    #     record['data/TFRecords/0099.mp4'], out_type=tf.float32)
    # print(type(tensor))
    dataset = tf.data.Dataset.from_tensor_slices(
        record).padded_batch(batch_size=32)
    it = iter(dataset)
    print(next(it).shape)
