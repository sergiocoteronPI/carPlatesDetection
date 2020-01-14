
import tensorflow as tf
from tensorflow import keras

def conv2d(inputs, f = 32, k = (3,3), s = 1, activation=None, padding = 'valid'):

    return tf.keras.layers.Conv2D(filters = f, kernel_size = k ,strides=(s, s),
                                  padding=padding,
                                  activation=activation,
                                  kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(inputs)
    
def leaky_relu(inputs, alpha = 0.2):
    
    return tf.keras.layers.LeakyReLU()(inputs)

def dropout(inputs, keep_prob):

    return tf.keras.layers.Dropout(keep_prob)(inputs)

def Flatten(inputs):
    
    return tf.keras.layers.Flatten()(inputs)

def Dense(inputs, units = 1024, use_bias = True, activation = None):
    
    return tf.keras.layers.Dense(units,activation=activation,use_bias=True,)(inputs)

def batch_norm(inputs):
    
    return tf.keras.layers.BatchNormalization(axis=-1,
                                              momentum=0.99,
                                              epsilon=0.001,
                                              center=True,
                                              scale=True,
                                              beta_initializer='zeros',
                                              gamma_initializer='ones',
                                              moving_mean_initializer='zeros',
                                              moving_variance_initializer='ones')(inputs)

def dense_layer(input_, reduccion, agrandamiento):

    dl_1 = conv2d(inputs = input_, f = reduccion, k = (1,1), s = 1)
    dl_1 = conv2d(inputs = dl_1, f = agrandamiento, k = (3,3), s = 1, padding = 'same')
    dl_1 = leaky_relu(tf.keras.layers.concatenate([input_, dl_1]))

    dl_2 = conv2d(inputs = dl_1, f = reduccion, k = (1,1), s = 1)
    dl_2 = conv2d(inputs = dl_2, f = agrandamiento, k = (3,3), s = 1, padding = 'same')
    dl_1 = leaky_relu(tf.keras.layers.concatenate([dl_1, dl_2]))

    dl_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=None,padding='valid')(dl_1)

    dl_2 = conv2d(inputs = dl_1, f = reduccion, k = (1,1), s = 1)
    dl_2 = conv2d(inputs = dl_2, f = agrandamiento, k = (3,3), s = 1, padding = 'same')
    dl_1 = leaky_relu(tf.keras.layers.concatenate([dl_1, dl_2]))

    dl_2 = conv2d(inputs = dl_1, f = reduccion, k = (1,1), s = 1)
    dl_2 = conv2d(inputs = dl_2, f = agrandamiento, k = (3,3), s = 1, padding = 'same')
    dl_1 = leaky_relu(tf.keras.layers.concatenate([dl_1, dl_2]))

    return dl_1


    def expit_tensor(x):
    return 1. / (1. + tf.exp(-tf.clip_by_value(x,-10,10)))

def calc_iou(boxes1, boxes2):

    boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                       boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                       boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                       boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
    boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

    boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                       boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                       boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                       boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
    boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

    lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
    rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

    intersection = tf.maximum(0.0, rd - lu)
    inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

    square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
              (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
    square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
              (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

    union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)