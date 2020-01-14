
import tensorflow as tf
from tensorflow import keras

from classMatDetec import classMatDetec
from utilsNeuralNetwork import *

import numpy as np
        
def lossFunction(yTrue, yPred):
    
    sprob = 1 #Coeficiente probabilidad clase
    sconf = 1 #Coeficiente objeto
    snoob = 0.25 #Coeficiente no objeto
    scoor = 5 #Coeficiente coordenadas
    
    H, W = classMatDetec.H, classMatDetec.W
    B, C = classMatDetec.B, classMatDetec.C
    
    anchors = classMatDetec.anchors

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

def neuralNetwork():

    x = tf.keras.Input(shape=(classMatDetec.dim_fil,classMatDetec.dim_col,3), name='input_layer')

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

    h_c1 = conv2d(inputs = h_c1, f = classMatDetec.B*(4+1+classMatDetec.C), k = (1,1), s = 1)

    model = tf.keras.Model(inputs=x, outputs=h_c1)

    return model, h_c1
