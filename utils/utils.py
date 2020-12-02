import tensorflow as tf
import matplotlib.pyplot as plt

def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float32)
    # shuffle points
    # points = tf.random.shuffle(points)
    return points, label


def decay_schedule(epoch, lr):
    # decay LR every 10 epochs
    if (epoch % 10 == 0) and (epoch != 0):
        lr = lr / 2
        print(f'Scheduled learning rate: {lr} on epoch {epoch}')
    return lr


def visualize_training(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for mean IOU
    plt.plot(history.history['m_iou'])
    plt.plot(history.history['val_m_iou'])
    plt.title('model mean IOU')
    plt.ylabel('mean IOU')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()