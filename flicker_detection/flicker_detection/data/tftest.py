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

    example_path = "TFRecords/0060.mp4.tfrecords"
    np.random.seed(0)
    ds = tf.data.TFRecordDataset(example_path).map(
        lambda byte: decode_fn(byte, "0060.mp4"))
    record = ds.get_single_element()
    print(record.shape)
    dataset = tf.data.Dataset.from_tensor_slices(
        record).padded_batch(batch_size=32)
    for batch in dataset:
        print(batch.shape)
