import tensorflow as tf


class Losses:

    @staticmethod
    def rmsle(y_true, y_pred):
        """Loss function"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.math.sqrt(
            tf.reduce_mean(
                tf.math.squared_difference(tf.math.log1p(y_pred),
                                           tf.math.log1p(y_true))
            )
        )
