
import numpy as np
import cv2
import os

import tensorflow as tf
import math
import sys

from random import shuffle

from classMatDetec import classMatDetec
from neuralNetwork import neuralNetwork, lossFunction

from auxiliarTrain import leerDatosTxt, cargarLote

### El usuario a lo mejor quiere cambiar donde se encuentras las imágenes y txt o algún otro parámetro del entrenamiento. ###
classMatDetec.cambiarParametro()

class soloNombres:
    def __init__(self, imageLabelNomb):
        self.imageLabelNomb = imageLabelNomb
            
imageLabelNomb = leerDatosTxt(ruta = classMatDetec.rpe)

shuffle(imageLabelNomb)
sn = soloNombres(imageLabelNomb)

if os.path.exists(classMatDetec.h5):

    #model = tf.keras.models.load_model(classMatDetec.h5, custom_objects={'loss_function': loss_function})

    model, h_out = neuralNetwork()
    model.compile(loss=lossFunction,optimizer=tf.keras.optimizers.Adam(lr = 0.001))
    #model.compile(loss=loss_function,optimizer=tf.keras.optimizers.RMSprop(lr=0.001,rho=0.9,epsilon=None,decay=0.0))
    model.load_weights(classMatDetec.h5)
    
else:

    model, h_out = neuralNetwork()
    #model.compile(loss=loss_function,optimizer=tf.keras.optimizers.Adam(lr = 0.001))
    model.compile(loss=lossFunction,optimizer=tf.keras.optimizers.RMSprop(lr=0.001,rho=0.9,epsilon=None,decay=0.0))

print('')
print(model.summary())
print('')

class MY_Generator(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):

        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_x = sn.imageLabelNomb[idx * self.batch_size:(idx + 1) * self.batch_size]
        #self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        image_train = []
        _yTrue = []
        
        for name in batch_x:
            try:
                _imagen_train, def_yTrue = cargarLote(classMatDetec, [name])
            except:
                continue

            if _imagen_train == []:
                continue
            
            image_train.append(_imagen_train[0])
            _yTrue.append(def_yTrue[0])

        return np.array(image_train), np.array(_yTrue)

##### =================================================================================== #####
my_training_batch_generator = MY_Generator(sn.imageLabelNomb, None, classMatDetec.batch_size)
##### =================================================================================== #####

def preparar_unlote(image_filenames, batch_size):

    batch_x = image_filenames[:batch_size]
        
    image_train = []
    _yTrue = []
    
    for name in batch_x:
        try:
            _imagen_train, def_yTrue = cargarLote(classMatDetec, [name])
        except:
            continue

        if _imagen_train == []:
            continue
        
        image_train.append(_imagen_train[0])
        _yTrue.append(def_yTrue[0])

    return np.array(image_train), np.array(_yTrue)

x_train_lote, y_train_lote = preparar_unlote(sn.imageLabelNomb, classMatDetec.batch_size)
model.fit(x_train_lote, y_train_lote, verbose=1)

while True:

    try:
        
        model.fit_generator(generator=my_training_batch_generator,
                            #validation_data=validation_generator,
                            steps_per_epoch= int(len(sn.imageLabelNomb) / classMatDetec.batch_size),
                            epochs=5,
                            verbose=1,
                            use_multiprocessing=True,
                            workers=3,
                            max_queue_size=10)

        print('')
        print(' ===== salvando modelo =====')
        print('')
                
        tf.keras.models.save_model(model, classMatDetec.h5)

        shuffle(sn.imageLabelNomb)

    except:break
