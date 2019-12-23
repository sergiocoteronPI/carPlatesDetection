
import numpy as np
import cv2
import os

import tensorflow as tf
import math
import sys

from random import shuffle

from clase_super_importante import self_

from auxiliar_train import leer_datos_text, programa_para_cargar_lote
from auxiliar_1 import postprocess

from ayuda import eliminar_elementos
from mark_1 import mark1, loss_function

class solo_nombres:
    def __init__(self, image_label_nomb):
        self.image_label_nomb = image_label_nomb
            
image_label_nomb = leer_datos_text(ruta = self_.rpe)

shuffle(image_label_nomb)
sn = solo_nombres(image_label_nomb)

if os.path.exists(self_.h5):

    #model = tf.keras.models.load_model(self_.h5, custom_objects={'loss_function': loss_function})

    model, h_out = mark1(self_)
    model.compile(loss=loss_function,optimizer=tf.keras.optimizers.Adam(lr = 0.001))
    #model.compile(loss=loss_function,optimizer=tf.keras.optimizers.RMSprop(lr=0.001,rho=0.9,epsilon=None,decay=0.0))
    model.load_weights(self_.h5)
    
else:

    model, h_out = mark1(self_)
    #model.compile(loss=loss_function,optimizer=tf.keras.optimizers.Adam(lr = 0.001))
    model.compile(loss=loss_function,optimizer=tf.keras.optimizers.RMSprop(lr=0.001,rho=0.9,epsilon=None,decay=0.0))

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

        batch_x = sn.image_label_nomb[idx * self.batch_size:(idx + 1) * self.batch_size]
        #self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        image_train = []
        _yTrue = []
        
        for name in batch_x:
            try:
                _imagen_train, def_yTrue = programa_para_cargar_lote(self_, [name])
            except:
                continue

            if _imagen_train == []:
                continue
            
            image_train.append(_imagen_train[0])
            _yTrue.append(def_yTrue[0])

        return np.array(image_train), np.array(_yTrue)

##### =================================================================================== #####
my_training_batch_generator = MY_Generator(sn.image_label_nomb, None, self_.batch_size)
##### =================================================================================== #####

def preparar_unlote(image_filenames, batch_size):

    batch_x = image_filenames[:batch_size]
        
    image_train = []
    _yTrue = []
    
    for name in batch_x:
        try:
            _imagen_train, def_yTrue = programa_para_cargar_lote(self_, [name])
        except:
            continue

        if _imagen_train == []:
            continue
        
        image_train.append(_imagen_train[0])
        _yTrue.append(def_yTrue[0])

    return np.array(image_train), np.array(_yTrue)

x_train_lote, y_train_lote = preparar_unlote(sn.image_label_nomb, self_.batch_size)
model.fit(x_train_lote, y_train_lote, verbose=1)

while True:

    try:
        
        model.fit_generator(generator=my_training_batch_generator,
                            #validation_data=validation_generator,
                            steps_per_epoch= int(len(sn.image_label_nomb) / self_.batch_size),
                            epochs=5,
                            verbose=1,
                            use_multiprocessing=True,
                            workers=3,
                            max_queue_size=10)

        print('')
        print(' ===== salvando modelo =====')
        print('')
                
        tf.keras.models.save_model(model, self_.h5)

        shuffle(sn.image_label_nomb)

    except:break

