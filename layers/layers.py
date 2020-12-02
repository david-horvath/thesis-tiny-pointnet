import numpy as np
import tensorflow as tf
from keras.layers import Layer
from keras.layers import Conv1D, Dense
from keras.layers import BatchNormalization


class ConvLayer(Layer):
    def __init__(self, filters, kernel_size, strides, padding='valid', activation=None,
                 apply_bn=False, bn_momentum=0.99, **kwargs):
        super(ConvLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn_momentum = bn_momentum
        self.conv = Conv1D(filters, kernel_size, strides=strides, padding=padding,
                           activation=activation, use_bias=not apply_bn)
        if apply_bn:
            self.bn = BatchNormalization(momentum=bn_momentum, fused=False)

    def call(self, x, training=None):
        x = self.conv(x)
        if self.apply_bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(ConvLayer, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'apply_bn': self.apply_bn,
            'bn_momentum': self.bn_momentum})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DenseLayer(Layer):
    def __init__(self, units, activation=None, apply_bn=False, bn_momentum=0.99, **kwargs):
        super(DenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn_momentum = bn_momentum
        self.dense = Dense(units, activation=activation, use_bias=not apply_bn)
        if apply_bn:
            self.bn = BatchNormalization(momentum=bn_momentum, fused=False)

    def call(self, x, training=None):
        x = self.dense(x)
        if self.apply_bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(DenseLayer, self).get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'apply_bn': self.apply_bn,
            'bn_momentum': self.bn_momentum})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TNet(Layer):
    def __init__(self, add_regularization=False, bn_momentum=0.99, **kwargs):
        super(TNet, self).__init__(**kwargs)
        self.add_regularization = add_regularization
        self.bn_momentum = bn_momentum
        self.conv0 = ConvLayer(16, 1, strides=1, bn_momentum=bn_momentum)
        self.conv1 = ConvLayer(32, 1, strides=1, bn_momentum=bn_momentum)
        self.conv2 = ConvLayer(256, 1, strides=1, bn_momentum=bn_momentum)
        self.fc0 = DenseLayer(128, activation=tf.nn.relu, apply_bn=True, bn_momentum=bn_momentum)
        self.fc1 = DenseLayer(64, activation=tf.nn.relu, apply_bn=True, bn_momentum=bn_momentum)

    def build(self, input_shape):
        self.K = input_shape[-1]

        self.w = self.add_weight(shape=(64, self.K ** 2), initializer=tf.zeros_initializer,
                                 trainable=True, name='w')
        self.b = self.add_weight(shape=(self.K, self.K), initializer=tf.zeros_initializer,
                                 trainable=True, name='b')

        # Initialize bias with identity
        I = tf.constant(np.eye(self.K), dtype=tf.float32)
        self.b = tf.math.add(self.b, I)

    def call(self, x, training=None):
        input_x = x  # BxNxK

        # Embed to higher dim
        x = tf.expand_dims(input_x, axis=2)  # BxNx1xK
        x = self.conv0(x, training=training)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = tf.squeeze(x, axis=2)  # BxNx256

        # Global features
        x = tf.reduce_max(x, axis=1)  # Bx256

        # Fully-connected layers
        x = self.fc0(x, training=training)  # Bx128
        x = self.fc1(x, training=training)  # Bx64

        # Convert to KxK matrix to matmul with input
        x = tf.expand_dims(x, axis=1)  # Bx1x64
        x = tf.matmul(x, self.w)  # Bx1xK^2
        x = tf.squeeze(x, axis=1)
        x = tf.reshape(x, (-1, self.K, self.K))

        # Add bias term (initialized to identity matrix)
        x += self.b

        # Add regularization
        if self.add_regularization:
            eye = tf.constant(np.eye(self.K), dtype=tf.float32)
            x_xT = tf.matmul(x, tf.transpose(x, perm=[0, 2, 1]))
            reg_loss = tf.nn.l2_loss(eye - x_xT)
            self.add_loss(1e-3 * reg_loss)

        return tf.matmul(input_x, x)

    def get_config(self):
        config = super(TNet, self).get_config()
        config.update({
            'add_regularization': self.add_regularization,
            'bn_momentum': self.bn_momentum})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
