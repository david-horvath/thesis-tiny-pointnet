import tensorflow as  tf
from keras.regularizers import Regularizer


class OrthogonalRegularizer(Regularizer):
    def __init__(self, num_features, l2_reg=0.001):
        self.num_features = num_features
        self.l2_reg = l2_reg
        self.eye = tf.eye(num_features)

    def __call__(self, inputs):
        inputs = tf.reshape(inputs, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(inputs, inputs, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2_reg * tf.square(xxt - self.eye))

    #  Enable serialization
    def get_config(self):
        return {
            'num_features': self.num_features,
            'l2_reg': self.l2_reg
        }

    # Enable deserialization
    def from_config(self, config):
        return self(**config)
