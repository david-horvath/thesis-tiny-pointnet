import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import Softmax
from keras.layers import Concatenate
from keras.layers import Dropout
from layers.layers import TNet, ConvLayer


def get_model(num_points, num_classes, bn_momentum=.99):
    pt_cloud = Input(shape=(num_points, 3), dtype=tf.float32, name='pt_cloud')  # BxNx3

    # Input transformer (B x N x 3 -> B x N x 3)
    pt_cloud_transform = TNet(bn_momentum=bn_momentum)(pt_cloud)

    # Embed to 64-dim space (B x N x 3 -> B x N x 32)
    pt_cloud_transform = tf.expand_dims(pt_cloud_transform, axis=2)  # for weight-sharing of conv

    hidden_32 = ConvLayer(32, 1, strides=1, activation=tf.nn.relu, apply_bn=True,
                          bn_momentum=bn_momentum)(pt_cloud_transform)

    embed_32 = ConvLayer(32, 1, strides=1, activation=tf.nn.relu, apply_bn=True,
                         bn_momentum=bn_momentum)(hidden_32)

    embed_32 = tf.squeeze(embed_32, axis=2)

    # Feature transformer (B x N x 32 -> B x N x 32)
    embed_32_transform_t = TNet(bn_momentum=bn_momentum, add_regularization=True)(embed_32)

    # Embed to 256-dim space (B x N x 32 -> B x N x 256)
    embed_32_transform = tf.expand_dims(embed_32_transform_t, axis=2)

    hidden_32 = ConvLayer(32, 1, strides=1, activation=tf.nn.relu, apply_bn=True,
                          bn_momentum=bn_momentum)(embed_32_transform)

    hidden_64 = ConvLayer(64, 1, strides=1, activation=tf.nn.relu, apply_bn=True,
                          bn_momentum=bn_momentum)(hidden_32)

    embed_256 = ConvLayer(256, 1, strides=1, activation=tf.nn.relu, apply_bn=True,
                          bn_momentum=bn_momentum)(hidden_64)

    embed_256 = tf.squeeze(embed_256, axis=2)

    # Global feature vector (B x N x 256 -> B x 256)
    global_descriptor = tf.reduce_max(embed_256, axis=1)

    global_descriptor = tf.expand_dims(global_descriptor, axis=1)

    global_descriptor = tf.tile(global_descriptor, [1, num_points, 1])

    # Segmentation block
    concatenate = Concatenate()([embed_32_transform_t, global_descriptor])

    hidden_128 = ConvLayer(128, 1, strides=1, activation=tf.nn.relu, apply_bn=True,
                           bn_momentum=bn_momentum)(concatenate)

    hidden_128 = Dropout(rate=0.5)(hidden_128)

    hidden_64 = ConvLayer(64, 1, strides=1, activation=tf.nn.relu, apply_bn=True,
                          bn_momentum=bn_momentum)(hidden_128)

    hidden_64 = Dropout(rate=0.5)(hidden_64)

    embed_32 = ConvLayer(32, 1, strides=1, activation=tf.nn.relu, apply_bn=True,
                         bn_momentum=bn_momentum)(hidden_64)

    segmentation_output = Conv1D(filters=num_classes, kernel_size=1, strides=1, padding='valid')(embed_32)

    logits = Softmax()(segmentation_output)

    return Model(inputs=pt_cloud, outputs=logits, name='TinyPointNet')


if __name__ == '__main__':
    model = get_model(num_points=4096, num_classes=13)

    model.summary()
