
import tensorflow as tf
from tensorflow.keras import backend as K

"""
keras metrics api:
https://keras.io/api/metrics/
custom sensitivity specificity:
https://stackoverflow.com/questions/55640149/error-in-keras-when-i-want-to-calculate-the-sensitivity-and-specificity
custom auc:
https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
"""


class Metrics():
    def __init__(self):
        super.__init__()

    @staticmethod
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall_keras = true_positives / (possible_positives + K.epsilon())
        return recall_keras

    @staticmethod
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision_keras = true_positives / (predicted_positives + K.epsilon())
        return precision_keras

    @staticmethod
    def specificity(y_true, y_pred):
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
        return tn / (tn + fp + K.epsilon())

    @staticmethod
    def negative_predictive_value(y_true, y_pred):
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
        return tn / (tn + fn + K.epsilon())

    @staticmethod
    def f1(y_true, y_pred):
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        return 2 * ((p * r) / (p + r + K.epsilon()))

    # TODO

    # def auroc(y_true, y_pred):
    #     """
    #     https://www.codegrepper.com/code-examples/python/auc+callback+keras
    #     """
    #     if tf.math.count_nonzero(y_true) == 0:
    #         return tf.cast(0.0, tf.float32)
    #     return tf.numpy_function(roc_auc_score, (y_true, y_pred), tf.float32)

    @staticmethod
    def fbeta(y_true, y_pred, beta=2):
        y_pred = K.clip(y_pred, 0, 1)

        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
        fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)), axis=1)
        fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        num = (1 + beta ** 2) * (p * r)
        den = (beta ** 2 * p + r + K.epsilon())
        return K.mean(num / den)

    @staticmethod
    def matthews_correlation_coefficient(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
        fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

        num = tp * tn - fp * fn
        den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        return num / K.sqrt(den + K.epsilon())

    @staticmethod
    def equal_error_rate(y_true, y_pred):
        n_imp = tf.math.count_nonzero(
            tf.equal(y_true, 0), dtype=tf.float32) + tf.constant(K.epsilon())
        n_gen = tf.math.count_nonzero(
            tf.equal(y_true, 1), dtype=tf.float32) + tf.constant(K.epsilon())

        scores_imp = tf.boolean_mask(y_pred, tf.equal(y_true, 0))
        scores_gen = tf.boolean_mask(y_pred, tf.equal(y_true, 1))

        loop_vars = (tf.constant(0.0), tf.constant(1.0), tf.constant(0.0))
        def cond(t, fpr, fnr): return tf.greater_equal(fpr, fnr)

        def body(t, fpr, fnr): return (
            t + 0.001,
            tf.divide(tf.math.count_nonzero(tf.greater_equal(
                scores_imp, t), dtype=tf.float32), n_imp),
            tf.divide(tf.math.count_nonzero(
                tf.less(scores_gen, t), dtype=tf.float32), n_gen)
        )
        t, fpr, fnr = tf.while_loop(cond, body, loop_vars, back_prop=False)
        eer = (fpr + fnr) / 2

        return eer
