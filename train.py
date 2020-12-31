import argparse
import os

import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.utils import np_utils
from datetime import datetime

from dataset.provider import dataset
from metrics.mIOU import mIOU
from model.model import get_model
from utils.utils import augment, decay_schedule, visualize_training


# Parse arguments
parser = argparse.ArgumentParser(allow_abbrev=False, description='Training script.')
parser.add_argument('--num_points',
                    action='store',
                    type=int,
                    default=4096,
                    help='Number of points.')
parser.add_argument('--num_classes',
                    action='store',
                    type=int,
                    default=13,
                    help='Number of segmentation classes.')
parser.add_argument('--batch_size',
                    action='store',
                    type=int,
                    default=64,
                    help='Batch size.')
parser.add_argument('--epochs',
                    action='store',
                    type=int,
                    default=100,
                    help='Number of training epochs.')
parser.add_argument('--lr',
                    action='store',
                    type=float,
                    default=1e-3,
                    help='Learning rate.')
parser.add_argument('--weights',
                    action='store',
                    type=str,
                    help='Pretrained model weights path.')
parser.add_argument('--dataset',
                    action='store',
                    type=str,
                    default='vkitti3d',
                    help='Dataset [ vkitti3d | s3dis ].')
parser.add_argument('--augment',
                    action='store',
                    type=bool,
                    default=True,
                    help='Data augmentation flag.')
parser.add_argument('--eval',
                    action='store',
                    type=bool,
                    default=True,
                    help='Evaulate the model.')
config = parser.parse_args()


# Define global required folder paths
now = datetime.now()
date = now.strftime('%Y-%m-%d_%Hh%Mm%Ss')

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DIST_PATH = BASE_PATH + f'/dist_{date}'
LOGS_DIR = DIST_PATH + '/logs'
SAVED_MODELS_DIR = DIST_PATH + '/saved_models'
SAVED_WEIGHTS_DIR = DIST_PATH + '/saved_weights'

# Create required folders
if not os.path.exists(DIST_PATH):
    os.mkdir(DIST_PATH)

if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)

if not os.path.exists(SAVED_MODELS_DIR):
    os.mkdir(SAVED_MODELS_DIR)

if not os.path.exists(SAVED_WEIGHTS_DIR):
    os.mkdir(SAVED_WEIGHTS_DIR)


# Generate tensorflow train and validation dataset, if config.eval flag is true generate test dataset too
train_data, train_label, val_data, val_label, test_data, test_label = dataset(dataset_name=config.dataset)

one_hot_train_label = np_utils.to_categorical(train_label, config.num_classes)
one_hot_val_label = np_utils.to_categorical(val_label, config.num_classes)

train_dataset = tf.data.Dataset.from_tensor_slices((train_data, one_hot_train_label))
val_dataset = tf.data.Dataset.from_tensor_slices((val_data, one_hot_val_label))
test_dataset = None

if config.augment:
    train_dataset = train_dataset.repeat(4).shuffle(len(train_data)).map(augment).batch(config.batch_size,
                                                                                        drop_remainder=True)
else:
    train_dataset = train_dataset.repeat(4).shuffle(len(train_data)).batch(config.batch_size, drop_remainder=True)

val_dataset = val_dataset.repeat(4).shuffle(len(val_data)).batch(config.batch_size, drop_remainder=True)

if config.eval:
    one_hot_test_label = np_utils.to_categorical(test_label, config.num_classes)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, one_hot_test_label))
    test_dataset = test_dataset.repeat(4).shuffle(len(test_data)).batch(config.batch_size, drop_remainder=True)

# Get Tiny-PointNet model
model = get_model(num_points=config.num_points, num_classes=config.num_classes)

# Display model architecture
model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=config.lr),
              loss=keras.losses.categorical_crossentropy,
              metrics=[
                  'accuracy',
                  mIOU(num_classes=config.num_classes)
              ])

# Define callbacks
checkpoint = ModelCheckpoint(
    filepath=SAVED_WEIGHTS_DIR + '/tiny-pointnet-weights-epoch_{epoch:02d}-loss_{loss:.4f}-accuracy_{accuracy:.4f}.hdf5',
    monitor='loss',
    verbose=1,
    save_best_only=True
)

early_stopping = EarlyStopping(
    monitor='loss',
    patience=5,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.2,
    patience=5,
    min_lr=0,
    verbose=1
)

lr_scheduler = LearningRateScheduler(decay_schedule, verbose=1)

tensorboard = TensorBoard(log_dir=LOGS_DIR + f'/tiny-pointnet-{config.dataset}')

callbacks = [
    checkpoint,
    early_stopping,
    reduce_lr,
    lr_scheduler,
    tensorboard
]


# Train the model
history = model.fit(train_dataset,
                    batch_size=config.batch_size,
                    epochs=config.epochs,
                    validation_data=val_dataset,
                    callbacks=callbacks,
                    shuffle=True)

# Save the model and model weights
model.save(SAVED_MODELS_DIR + f'/Tiny-PointNet-{config.dataset}')
model.save_weights(SAVED_WEIGHTS_DIR + f'/Tiny-PointNet-weights-final-{config.dataset}.hdf5')

# Visualize loss, accuracy and mean IOU
visualize_training(history)

# Evaluate the model if eval flag is true and test dataset is exists
if config.eval and test_dataset is not None:
    loss, acc, _ = model.evaluate(test_dataset, verbose=1)
    print(f'Test loss: {loss}')
    print(f'Test accuracy: {acc}')
