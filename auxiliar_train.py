
import numpy as np
import cv2
import os

from clase_super_importante import self_

from random import shuffle
from copy import deepcopy

font = cv2.FONT_HERSHEY_SIMPLEX

def leer_datos_text(ruta):

    image_label_nomb = []

    for ruta, _, ficheros in os.walk(ruta):
        for nombre_fichero in ficheros:
            rut_comp = os.path.join(ruta, nombre_fichero)
            if rut_comp.endswith("txt"):
                image_label_nomb.append(rut_comp)
    return image_label_nomb

def retocar(self, img, cosico):
    
    zeros = np.zeros([self.dim_fil,self.dim_col,3])
    im_sha_1, im_sha_2, _ = img.shape
    
    if im_sha_1 >= self.dim_fil:
        if im_sha_2 >= self.dim_col:
            zeros = cv2.resize(img,(self.dim_col,self.dim_fil))
            for obj in cosico:
                obj[2],obj[1],obj[4],obj[3] = int(obj[2]*self.dim_fil/im_sha_1), int(obj[1]*self.dim_col/im_sha_2), int(obj[4]*self.dim_fil/im_sha_1), int(obj[3]*self.dim_col/im_sha_2)
        else:
            zeros[:,0:im_sha_2,:] = cv2.resize(img,(im_sha_2,self.dim_fil))
            for obj in cosico:
                obj[2],obj[4] = int(obj[2]*self.dim_fil/im_sha_1), int(obj[4]*self.dim_fil/im_sha_1)
    elif im_sha_2 >= self.dim_col:
        zeros[0:im_sha_1,:,:] = cv2.resize(img,(self.dim_col,im_sha_1))
        for obj in cosico:
            obj[1],obj[3] = int(obj[1]*self.dim_col/im_sha_2), int(obj[3]*self.dim_col/im_sha_2)
    else:
        zeros[0:im_sha_1, 0:im_sha_2,:] = img

    return zeros, cosico

def dibujar_imagen_y_coordenadas(self_, image, final_bbox):

    for box in final_bbox:
        mess, left, top, right, bot = box[0], int(box[1]), int(box[2]), int(box[3]), int(box[4])
        
        cv2.rectangle(image,(left, top), (right, bot),self_.colors[self_.labels.index(mess)], 2)
            
        if top - 16 > 0:
            cv2.rectangle(image,(left-1, top - 16), (left + (len(mess))*5*2-1, top),self_.colors[self_.labels.index(mess)], -1)
            cv2.putText(image,mess ,(left, top - 4), font, 0.45,(0,0,0),0,cv2.LINE_AA)

        else:
            cv2.rectangle(image,(left-1, top), (left + (len(mess))*5*2-1, top+16),self_.colors[self_.labels.index(mess)], -1)
            cv2.putText(image,mess,(left, top + 12), font, 0.45,(0,0,0),0,cv2.LINE_AA)

def visualiza(self_):
#if True:
    
    image_label_nomb = leer_datos_text(ruta = self_.rpe)
    shuffle(image_label_nomb)

    for name in image_label_nomb:
        """ Abrimos los .txt y leemos nombre de la imagen tamaño y las cajas. Le pasamos estos datos _batch """
        vector = []
        with open(name, 'r') as f:
            for line in f:
                linea = line.rstrip('\n').split(',')
                vector.append(linea)

        if vector == []:
            continue

        image = cv2.imread(self_.rpi + vector[0][0])

        bboxes = []
        for mini_vector in vector:
            bboxes.append([float(mini_vector[1]),float(mini_vector[2]),float(mini_vector[3]),float(mini_vector[4])])

        final_bbox = []
        coooount = 0
        for box in bboxes:
            if int(box[0]) != 0 or int(box[1]) != 0 or int(box[2]) != 0 or int(box[3]) != 0:
                final_bbox.append([vector[coooount][1], box[0], box[1], box[2], box[3]])
            coooount += 1

        for box in final_bbox:
            mess, left, top, right, bot = box[0], int(box[1]), int(box[2]), int(box[3]), int(box[4])
            
            cv2.rectangle(image,(left, top), (right, bot),self_.colors[self_.labels.index(mess)], 2)
                
            if top - 16 > 0:
                cv2.rectangle(image,(left-1, top - 16), (left + (len(mess))*5*2-1, top),self_.colors[self_.labels.index(mess)], -1)
                cv2.putText(image,mess ,(left, top - 4), font, 0.45,(0,0,0),0,cv2.LINE_AA)

            else:
                cv2.rectangle(image,(left-1, top), (left + (len(mess))*5*2-1, top+16),self_.colors[self_.labels.index(mess)], -1)
                cv2.putText(image,mess,(left, top + 12), font, 0.45,(0,0,0),0,cv2.LINE_AA)

        cv2.imshow('image', image)
        cv2.waitKey(0)

#Cosa para visualizar
#visualiza(self_)

def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def _batch(self, img ,allobj):

    shuffle(allobj)

    H, W = self.H, self.W
    B, C = self.B, self.C
    
    #anchors = self.anchors
    labels = self.labels
    
    cellx = 1. * self.dim_col / W
    celly = 1. * self.dim_fil / H

    y_true_etiqueta = np.zeros([H,W,B*(1+4+C)])
    for obj in allobj:

        if obj[0] not in labels:
            continue

        area = (obj[3]-obj[1])*(obj[4]-obj[2])
        
        centerx = .5*(obj[1]+obj[3]) #xmin, xmax
        centery = .5*(obj[2]+obj[4]) #ymin, ymax
        cx = centerx / cellx
        cy = centery / celly

        if cx >= W or cy >= H: return []
        
        if obj[1] < 1:
            obj[1] = 1
        if obj[2] < 1:
            obj[2] = 1
        if obj[3] > self.dim_col:
            obj[3] = self.dim_col - 1
        if obj[4] > self.dim_fil:
            obj[4] = self.dim_fil - 1
            
        obj[3] = float(obj[3]-obj[1]) / self.dim_col
        obj[4] = float(obj[4]-obj[2]) / self.dim_fil

        for cog in range(3,5):
            if obj[cog] < 0:
                obj[cog] = 0.001
        
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])
        
        obj[1] = cx - np.floor(cx) # centerx
        obj[2] = cy - np.floor(cy) # centery

        if area < 10000:
            r_b = 0
        elif 10000 <= area < 50000:
            r_b = 1
        else:
            r_b = 2

        if True: 
        #for r_b in range(B):
            
            numb_magic = int(np.floor(cy) * W + np.floor(cx))

            resto = int(numb_magic%W)
            el_otro = int((numb_magic - resto) / W)

            if y_true_etiqueta[el_otro, resto, B*4 + r_b] == 0:

                y_true_etiqueta[el_otro, resto, 4*r_b : 4*(r_b+1)] = obj[1:5]
                y_true_etiqueta[el_otro, resto, B*4 + r_b] = 1.
                y_true_etiqueta[el_otro, resto, B*5 + r_b*len(labels) + labels.index(obj[0])] = 1.
    
    return img/255 * 2 - 1, y_true_etiqueta

def agrandar_bboxes(self_, image, bboxes):

    H, W = self_.H, self_.W

    sha1, sha2, _ = image.shape

    cellx = 1. * sha2 / W
    celly = 1. * sha1 / H

    new_bboxes = deepcopy(bboxes)
    for box in bboxes:

        mess, left, top, right, bot = box[0], int(box[1]), int(box[2]), int(box[3]), int(box[4])

        centerx = .5*(left+right)
        centery = .5*(top+bot)
        cx = centerx / cellx
        cy = centery / celly
        
        ml_x, ml_y = (right - left)/2, (bot - top)/2

        for i, j in [(0,1), (1,1), (1,0), (0,-1), (-1,-1), (-1,0), (1,-1), (-1,1)]:
            
            if (cx + i) > 0 and (cx + i) < W and (cy + j) > 0 and (cy + j) < H:

                new_cx, new_cy = cx + i, cy + j

                if new_cx < cx:
                    new_cx += np.ceil(cx) - cx
                elif new_cx > cx:
                    new_cx -= cx  - np.floor(cx)

                
                if new_cy < cy:
                    new_cy += np.ceil(cy) - cy
                elif new_cy > cy:
                    new_cy -= cy  - np.floor(cy)

                new_left, new_right = int(new_cx*cellx - ml_x), int(new_cx*cellx + ml_x)
                new_top, new_bot = int(new_cy*celly - ml_y), int(new_cy*celly + ml_y)

                if iou([new_left, new_top, new_right, new_bot],[left, top, right, bot]) < 0.5:
                    continue

                new_bboxes.append([mess, new_left, new_top, new_right, new_bot])

    return new_bboxes
    
def leer_imagen_en_eg_o_color(nombre):

    if np.random.randint(10)%2:

        return cv2.imread(nombre)

    else:
        
        eg = cv2.imread(nombre, 0)
        largo, ancho = eg.shape

        eg_triple = np.zeros([largo, ancho,3])

        eg_triple[:,:,0] = eg
        eg_triple[:,:,1] = eg
        eg_triple[:,:,2] = eg

        return eg_triple.astype('uint8')

def retocar_imagen_y_coordenadas(imagen, bboxes):

    sha1, sha2, _ = imagen.shape

    if np.random.randint(10)%2:
        sha_y,sha_x,_= imagen.shape
        noise = np.random.rand(sha_y,sha_x,3)
        imagen = imagen + noise*np.random.randint(3,7)

    #Suavizar imagen
    if np.random.randint(10)%2:
        imagen = cv2.filter2D(imagen,-1,np.ones((5,5),np.float32)/25)

    #Difuminada
    if np.random.randint(10)%2:
        imagen = cv2.blur(imagen,(3,3))

    #flip
    if np.random.randint(10)%2:

        imagen = cv2.flip(imagen, 1)
        for box in bboxes:
            aux = sha2 - deepcopy(box[1])
            box[1] = sha2 - box[3]
            box[3] = aux

    #crop
    if np.random.randint(10)%2:

        box = bboxes[np.random.randint(len(bboxes))]
        
        new_x = np.random.randint(0,box[1])
        new_y = np.random.randint(0,box[2])

        new_w = np.random.randint(box[3], sha2)
        new_h = np.random.randint(box[4], sha1)

        newbboxes = []
        for box in bboxes:

            box[1] = box[1] - new_x
            box[2] = box[2] - new_y

            box[3] = box[3] - new_x
            box[4] = box[4] - new_y

            if box[1] >= 0:
                if box[2] >= 0:
                    newbboxes.append([box[0],box[1],box[2],box[3],box[4]])
                elif box[4] > 1:
                    box[2] = 1
                    newbboxes.append([box[0],box[1],box[2],box[3],box[4]])
            elif box[3] > 1:
                box[1] = 1
                newbboxes.append([box[0],box[1],box[2],box[3],box[4]])

        bboxes = newbboxes
        imagen = imagen[new_y:new_h, new_x:new_w, :]

    #rotate


    return imagen, bboxes

def programa_para_cargar_lote(self, nom_lot):

    im_train = []
    y_true_nnp = []

    for name in nom_lot:
        
        vector = []
        with open(name, 'r') as f:
            for line in f:
                linea = line.rstrip('\n').split(',')
                vector.append(linea)

        if vector == []:
            return [], []

        image = leer_imagen_en_eg_o_color(self_.rpi + vector[0][0])
        bboxes = []
        for mini_vector in vector:
            bboxes.append(["matricula", float(mini_vector[1]),float(mini_vector[2]),float(mini_vector[3]),float(mini_vector[4])])
        
        image, bboxes = retocar_imagen_y_coordenadas(image, bboxes)
        transformed_image, final_bbox = retocar(self, image,bboxes)

        final_bbox = agrandar_bboxes(self, transformed_image,final_bbox)

        im_de_out_batch, y_true_ = _batch(self, transformed_image ,final_bbox)
        
        im_train.append(im_de_out_batch)
        y_true_nnp.append(y_true_)

    return im_train, y_true_nnp

#Programa para visualizar la funcion cargar_lote
if False:
    image_label_nomb = leer_datos_text(ruta = self_.rpe)

    for name in image_label_nomb:

        vector = []
        with open(name, 'r') as f:
            for line in f:
                linea = line.rstrip('\n').split(',')
                vector.append(linea)

        image = leer_imagen_en_eg_o_color(self_.rpi + vector[0][0])
        image_original, _ = retocar(self_, deepcopy(image), [])

        bboxes = []
        for mini_vector in vector:
            bboxes.append(["matricula", float(mini_vector[1]),float(mini_vector[2]),float(mini_vector[3]),float(mini_vector[4])])
        
        image, bboxes = retocar_imagen_y_coordenadas(image, bboxes)
        image, bboxes = retocar(self_, image,bboxes)

        shuffle(bboxes)
        for box in bboxes:

            mess, left, top, right, bot = "matricula", int(box[1]), int(box[2]), int(box[3]), int(box[4])

            #print((right - left)*(bot - top))

            cv2.rectangle(image,(left, top), (right, bot),self_.colors[self_.labels.index(mess)], 2)
                    
            if top - 16 > 0:
                cv2.rectangle(image,(left-1, top - 16), (left + (len(mess))*5*2-1, top),self_.colors[self_.labels.index(mess)], -1)
                cv2.putText(image,mess ,(left, top - 4), font, 0.45,(0,0,0),0,cv2.LINE_AA)

            else:
                cv2.rectangle(image,(left-1, top), (left + (len(mess))*5*2-1, top+16),self_.colors[self_.labels.index(mess)], -1)
                cv2.putText(image,mess,(left, top + 12), font, 0.45,(0,0,0),0,cv2.LINE_AA)

        cv2.imshow('image', image.astype('uint8'))
        cv2.imshow('image_org', image_original.astype('uint8'))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

if False:

    H, W = self_.H, self_.W
    B, C = self_.B, self_.C
    
    image_label_nomb = leer_datos_text(ruta = self_.rpe)
    shuffle(image_label_nomb)

    for name in image_label_nomb:
        
        vector = []
        with open(name, 'r') as f:
            for line in f:
                linea = line.rstrip('\n').split(',')
                if linea[1] in self_.labels:
                    vector.append(linea)

        if vector == []:
            print("Ezebez")
            continue

        image = leer_imagen_en_eg_o_color(self_.rpi + vector[0][0])
        bboxes = []
        for mini_vector in vector:
            bboxes.append([mini_vector[1], float(mini_vector[2]),float(mini_vector[3]),float(mini_vector[4]),float(mini_vector[5])])
            
        image, bboxes = retocar_imagen_y_coordenadas(image, bboxes)
        image, bboxes = retocar(self_, image,bboxes)
        
        ###########################       VISUALIZACION DE NUEVA FORMA PARA ENTRENAR       ###########################
        sha1, sha2, _ = image.shape

        cellx = 1. * sha2 / W
        celly = 1. * sha1 / H


        tam_1, tam_2 = sha1/H, sha2/W
        for di_fi in range(H):
            for di_co in range(W):
                cv2.rectangle(image,(int(di_co*tam_2), int(di_fi*tam_1)), (int((1+di_co)*tam_2), int((1+di_fi)*tam_1)),(0,0,0), 1)
        
        for box in bboxes:
            mess, left, top, right, bot = box[0], int(box[1]), int(box[2]), int(box[3]), int(box[4])

            centerx = .5*(left+right) #xmin, xmax
            centery = .5*(top+bot) #ymin, ymax
            cx = centerx / cellx
            cy = centery / celly
            
            ml_x, ml_y = (right - left)/2, (bot - top)/2

            #punto_medio_x, punto_medio_y = int(left + (right - left)/2), int(top + (bot - top)/2)
            #cv2.circle(image,(punto_medio_x,punto_medio_y), 5, (255,0,0), -1)

            for i,j in [(0,1),(1,1),(1,0), (0,-1),(-1,-1),(-1,0), (1,-1), (-1,1)]:
                
                if (cx + i) > 0 and (cx + i) < W and (cy + j) > 0 and (cy + j) < H:

                    new_cx, new_cy = cx + i, cy + j

                    if new_cx < cx:
                        new_cx += np.ceil(cx) - cx
                    elif new_cx > cx:
                        new_cx -= cx  - np.floor(cx)

                    
                    if new_cy < cy:
                        new_cy += np.ceil(cy) - cy
                    elif new_cy > cy:
                        new_cy -= cy  - np.floor(cy)
                    

                    #cv2.circle(image,(int(new_cx*cellx),int(new_cy*celly)), 2, (255,0,255), -1)

                    new_left, new_right = int(new_cx*cellx - ml_x), int(new_cx*cellx + ml_x)
                    new_top, new_bot = int(new_cy*celly - ml_y), int(new_cy*celly + ml_y)

                    if iou([new_left, new_top, new_right, new_bot],[left, top, right, bot]) < 0.5:
                        continue

                    cv2.rectangle(image,(new_left, new_top),(new_right, new_bot),self_.colors[self_.labels.index(mess)], 1)

        ###########################       VISUALIZACION DE NUEVA FORMA PARA ENTRENAR       ###########################

        #for box in bboxes:
            #mess, left, top, right, bot = box[0], int(box[1]), int(box[2]), int(box[3]), int(box[4])
            
            cv2.rectangle(image,(left, top), (right, bot),self_.colors[self_.labels.index(mess)], 2)
                
            if top - 16 > 0:
                cv2.rectangle(image,(left-1, top - 16), (left + (len(mess))*5*2-1, top),self_.colors[self_.labels.index(mess)], -1)
                cv2.putText(image,mess ,(left, top - 4), font, 0.45,(0,0,0),0,cv2.LINE_AA)

            else:
                cv2.rectangle(image,(left-1, top), (left + (len(mess))*5*2-1, top+16),self_.colors[self_.labels.index(mess)], -1)
                cv2.putText(image,mess,(left, top + 12), font, 0.45,(0,0,0),0,cv2.LINE_AA)

            cv2.imshow('image', image.astype('uint8'))

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        #break