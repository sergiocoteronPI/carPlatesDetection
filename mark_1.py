
import tensorflow as tf
from tensorflow import keras

from clase_super_importante import self_ as self

import numpy as np

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
    """calculate ious
    Args:
        boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
        boxes2: 1-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
    Return:
        iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    """
    
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

    # calculate the left up point & right down point
    lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
    rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

    # intersection
    intersection = tf.maximum(0.0, rd - lu)
    inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

    # calculate the boxs1 square and boxs2 square
    square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
              (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
    square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
              (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

    union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)
        
def loss_function(yTrue, yPred):
    
    sprob = 1 #Coeficiente probabilidad clase
    sconf = 1 #Coeficiente objeto
    snoob = 0.25 #Coeficiente no objeto
    scoor = 5 #Coeficiente coordenadas
    
    H, W = self.H, self.W
    B, C = self.B, self.C
    
    anchors = self.anchors

    _coord = tf.reshape(yTrue[:,:,:,:B*4], [-1, H*W, B, 4])
    _confs = tf.reshape(yTrue[:,:,:,B*4:B*5], [-1, H*W, B])
    _probs = tf.reshape(yTrue[:,:,:,B*5:], [-1, H*W, B, C])

    _uno_obj = tf.reshape(tf.minimum(tf.reduce_sum(_confs, [2]), 1.0),[-1, H*W])

    net_out_coords = tf.reshape(yPred[:,:,:,:B*4], [-1, H*W, B, 4])
    net_out_confs = tf.reshape(yPred[:,:,:,B*4:B*5], [-1, H, W, B])
    net_out_probs = tf.reshape(yPred[:,:,:,B*5:], [-1, H, W, B, C])
                                                                                                                 
    #coords = tf.reshape(net_out_coords, [-1, H*W, B, 4])
    adjusted_coords_xy = expit_tensor(net_out_coords[:,:,:,0:2])
    adjusted_coords_wh = tf.sqrt(tf.exp(tf.clip_by_value(net_out_coords[:,:,:,2:4],-15,8))* np.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))
    adjusted_coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)
    
    x_yolo = tf.reshape(adjusted_coords_xy[:,:,:,0],[-1,H*W,B])
    y_yolo = tf.reshape(adjusted_coords_xy[:,:,:,1],[-1,H*W,B])
    w_yolo = tf.reshape(adjusted_coords_wh[:,:,:,0],[-1,H*W,B])
    h_yolo = tf.reshape(adjusted_coords_wh[:,:,:,1],[-1,H*W,B])
    
    adjusted_c = expit_tensor(net_out_confs)
    adjusted_c = tf.reshape(adjusted_c, [-1, H*W, B])
    
    adjusted_prob = expit_tensor(net_out_probs)
    adjusted_prob = tf.reshape(adjusted_prob,[-1, H*W, B, C])
    
    iou = calc_iou(tf.reshape(_coord, [-1, H, W, B, 4]), tf.reshape(adjusted_coords,[-1, H, W, B, 4]))
    best_box = tf.reduce_max(iou, 3, keepdims=True)
    best_box = tf.to_float(best_box)
    confs = tf.reshape(tf.cast((iou >= best_box), tf.float32),[-1,H*W,B]) * _confs

    coord_loss_xy = scoor*tf.reduce_mean(tf.reduce_sum(_confs*(tf.reshape(tf.square(x_yolo - _coord[:,:,:,0]) + tf.square(y_yolo - _coord[:,:,:,1]),[-1,H*W,B])),[1,2]))# + \
    coord_loss_wh = scoor*tf.reduce_mean(tf.reduce_sum(_confs*(tf.reshape(tf.square(w_yolo - _coord[:,:,:,2]) + tf.square(h_yolo - _coord[:,:,:,3]),[-1,H*W,B])),[1,2]))# + \
    
    conf_loss = sconf*tf.reduce_mean(tf.reduce_sum(_confs*tf.square(adjusted_c - confs),[1,2])) + \
                snoob*tf.reduce_mean(tf.reduce_sum((1.0 - _confs)*tf.square(adjusted_c - confs),[1,2]))
    
    class_loss = sprob*tf.reduce_mean(tf.reduce_sum(_uno_obj*tf.reduce_sum(tf.square(adjusted_prob - _probs),[2,3]),1))
    #class_loss = sprob*tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(adjusted_prob - _probs),[2,3]),1))

    loss = coord_loss_xy + coord_loss_wh + class_loss + conf_loss

    return loss#, class_loss, conf_loss, coord_loss

def mark1(self):

    x = tf.keras.Input(shape=(self.dim_fil,self.dim_col,3), name='input_layer')

    h_c1 = conv2d(inputs = x, f = 8, k = (3,3), s = 2, padding='same')
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=None,padding='valid')(x)
    h_c1 = tf.keras.layers.concatenate([pool1, h_c1])

    h_c1 = conv2d(inputs = h_c1, f = 16, k = (3,3), s = 2, padding='same')
    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=None,padding='valid')(pool1)
    h_c1 = leaky_relu(tf.keras.layers.concatenate([pool2, h_c1]))

    h_c1 = batch_norm(conv2d(inputs = h_c1, f = 32, k = (3,3), s = 2))

    h_c1 = dense_layer(h_c1, 16, 32)
    h_c1 = leaky_relu(batch_norm(conv2d(inputs = h_c1, f = 512, k = (3,3), s = 1)))

    h_c1 = dense_layer(h_c1, 32, 64)

    h_c1 = conv2d(inputs = h_c1, f = self.B*(4+1+self.C), k = (1,1), s = 1)

    model = tf.keras.Model(inputs=x, outputs=h_c1)

    return model, h_c1
