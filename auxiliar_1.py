
import numpy as np

import cv2
import os

from clase_super_importante import self_

font = cv2.FONT_HERSHEY_SIMPLEX

class BoundBox:
    def __init__(self, classes):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.box = int()
        self.lab = ''
        self.probs = float()#np.zeros((classes,))
        self.prob_obj = float()
        self.prob_class = float()
        #self.

def overlap_c(x1, w1 , x2 , w2):
    l1 = x1 - w1 /2.
    l2 = x2 - w2 /2.
    left = max(l1,l2)
    r1 = x1 + w1 /2.
    r2 = x2 + w2 /2.
    right = min(r1, r2)
    return right - left

def box_intersection_c(ax, ay, aw, ah, bx, by, bw, bh):
    w = overlap_c(ax, aw, bx, bw)
    h = overlap_c(ay, ah, by, bh)
    if w < 0 or h < 0: return 0
    area = w * h
    return area

def box_union_c(ax, ay, aw, ah, bx, by, bw, bh):
    i = box_intersection_c(ax, ay, aw, ah, bx, by, bw, bh)
    u = aw * ah + bw * bh -i
    return u

def box_iou_c(ax, ay, aw, ah, bx, by, bw, bh):
    return box_intersection_c(ax, ay, aw, ah, bx, by, bw, bh) / box_union_c(ax, ay, aw, ah, bx, by, bw, bh)

def expit_c(x):
    return 1/(1+np.exp(-np.clip(x,-10,10)))
    
def NMS(self, final_probs , final_bbox):

    labels = self.labels
    C = self.C
    
    boxes = []
    indices = []
  
    pred_length = final_bbox.shape[0]
    class_length = final_probs.shape[1]

    #Este bucle recorre el número de clases
    for class_loop in range(class_length):

        #Este otro bucle recorre H*W*B es decir el número de cuadrados por cajitas
        for index in range(pred_length):

            #Cuando la probabilidad es 0 entonces pasamos a la siguiente cosa de recorrer cuadrados y cajitas
            if final_probs[index,class_loop] == 0: continue

            #Si no comenzamos un bucle que recorra los indices desde index + 1 hasta pred_length
            for index2 in range(index+1,pred_length):
                #En caso de que las probabilidades vayan siendo 0 o que index sea igual a index2 se pasa al siguiente
                if final_probs[index2,class_loop] == 0: continue
                if index==index2 : continue

                #Se calcula la mejor caja
                if box_iou_c(final_bbox[index,0],final_bbox[index,1],final_bbox[index,2],final_bbox[index,3],final_bbox[index2,0],final_bbox[index2,1],final_bbox[index2,2],final_bbox[index2,3]) >= 0.1:
                    if final_probs[index2,class_loop] > final_probs[index, class_loop] :
                        final_probs[index, class_loop] = 0
                        break
                    final_probs[index2,class_loop]=0
            if index not in indices:

                bb=BoundBox(C)
                bb.x = final_bbox[index, 0]
                bb.y = final_bbox[index, 1]
                bb.w = final_bbox[index, 2]
                bb.h = final_bbox[index, 3]
                #bb.box = box_loop
                bb.lab = labels[class_loop]
                bb.probs = final_probs[index,class_loop]
                #bb.prob_obj = Bbox_pred[row, col, box_loop, 4]
                #bb.prob_class = Classes[row, col, class_loop]
                boxes.append(bb)

                """
                bb=BoundBox(class_length)
                bb.x = final_bbox[index, 0]
                bb.y = final_bbox[index, 1]
                bb.w = final_bbox[index, 2]
                bb.h = final_bbox[index, 3]
                bb.c = final_bbox[index, 4]
                bb.probs = np.asarray(final_probs[index,:])
                boxes.append(bb)
                """
                
                indices.append(index)
                
    return boxes

def box_constructor(self, net_out_in):

    """ Cargamos ahora los valores predeterminados en el programa principal a traves de self """
    threshold = self.threshold
    anchors = self.anchors

    H, W = self.H, self.W
    B, C = self.B, self.C
    
    #boxes = []

    #Bbox_pred = net_out_in[:,:,:,:B*5].reshape([H, W, B,(1+4)])
    #Classes = net_out_in[:,:,:,B*5:].reshape([H, W, B, C])

    Bbox_pred = net_out_in[:,:,:,:B*4].reshape([H, W, B,4])
    Conf_pred = net_out_in[:,:,:,B*4:B*5].reshape([H, W, B])
    Classes = net_out_in[:,:,:,B*5:].reshape([H, W, B, C])
    
    probs = np.zeros((H, W, B, C), dtype=np.float32)
    _Bbox_pred = np.zeros((H, W, B, 5), dtype=np.float32)
    
    for row in range(H):
        for col in range(W):
            for box_loop in range(B):

                Classes[row, col, box_loop, :] = expit_c(Classes[row, col, box_loop, :])
                if np.max(Classes[row, col, box_loop, :]) < threshold:
                    continue
            
                Conf_pred[row, col, box_loop,] = expit_c(Conf_pred[row, col, box_loop])
                if Conf_pred[row, col, box_loop] < threshold:
                    continue
                
                Bbox_pred[row, col, box_loop, 0] = (col + expit_c(Bbox_pred[row, col, box_loop, 0])) / W
                Bbox_pred[row, col, box_loop, 1] = (row + expit_c(Bbox_pred[row, col, box_loop, 1])) / H
                Bbox_pred[row, col, box_loop, 2] = np.exp(np.clip(Bbox_pred[row, col, box_loop, 2],-15,8)) * anchors[2 * box_loop + 0] / W
                Bbox_pred[row, col, box_loop, 3] = np.exp(np.clip(Bbox_pred[row, col, box_loop, 3],-15,8)) * anchors[2 * box_loop + 1] / H

                for class_loop in range(C):

                    #Nos permite seleccionar las clases que mostramos 20181120
                    if class_loop not in self.clases_visibles:
                        continue
                    
                    tempc = Classes[row, col, box_loop, class_loop] * Conf_pred[row, col, box_loop]
                    if(tempc > threshold):

                        probs[row, col, box_loop, class_loop] = tempc
                        _Bbox_pred[row, col, box_loop, 0] = Bbox_pred[row, col, box_loop, 0]
                        _Bbox_pred[row, col, box_loop, 1] = Bbox_pred[row, col, box_loop, 1]
                        _Bbox_pred[row, col, box_loop, 2] = Bbox_pred[row, col, box_loop, 2]
                        _Bbox_pred[row, col, box_loop, 3] = Bbox_pred[row, col, box_loop, 3]
                        _Bbox_pred[row, col, box_loop, 4] = Conf_pred[row, col, box_loop]
                        
    return NMS(self, np.ascontiguousarray(probs).reshape(H*W*B,C), np.ascontiguousarray(_Bbox_pred).reshape(H*W*B,5))#boxes#

def box_constructor_sin_nms(self, net_out_in):

    """ Cargamos ahora los valores predeterminados en el programa principal a traves de self """
    threshold = self.threshold
    labels = self.labels
    anchors = self.anchors

    H, W = self.H, self.W
    B, C = self.B, self.C
    
    boxes = []

    #Bbox_pred = net_out_in[:,:,:,:B*5].reshape([H, W, B,(1+4)])
    #Classes = net_out_in[:,:,:,B*5:].reshape([H, W, B, C])

    Bbox_pred = net_out_in[:,:,:,:B*4].reshape([H, W, B,4])
    Conf_pred = net_out_in[:,:,:,B*4:B*5].reshape([H, W, B])
    Classes = net_out_in[:,:,:,B*5:].reshape([H, W, B, C])
    
    #probs = np.zeros((H, W, B, C), dtype=np.float32)
    _Bbox_pred = np.zeros((H, W, B, 5), dtype=np.float32)
    
    for row in range(H):
        for col in range(W):
            for box_loop in range(B):

                Classes[row, col, box_loop, :] = expit_c(Classes[row, col, box_loop, :])
                if np.max(Classes[row, col, box_loop, :]) < threshold:
                    continue
            
                Conf_pred[row, col, box_loop,] = expit_c(Conf_pred[row, col, box_loop])
                if Conf_pred[row, col, box_loop] < threshold:
                    continue
                
                Bbox_pred[row, col, box_loop, 0] = (col + expit_c(Bbox_pred[row, col, box_loop, 0])) / W
                Bbox_pred[row, col, box_loop, 1] = (row + expit_c(Bbox_pred[row, col, box_loop, 1])) / H
                Bbox_pred[row, col, box_loop, 2] = np.exp(np.clip(Bbox_pred[row, col, box_loop, 2],-15,8)) * anchors[2 * box_loop + 0] / W
                Bbox_pred[row, col, box_loop, 3] = np.exp(np.clip(Bbox_pred[row, col, box_loop, 3],-15,8)) * anchors[2 * box_loop + 1] / H

                for class_loop in range(C):

                    #Nos permite seleccionar las clases que mostramos 20181120
                    if class_loop not in self.clases_visibles:
                        continue
                    
                    tempc = Classes[row, col, box_loop, class_loop] * Conf_pred[row, col, box_loop]
                    if(tempc > threshold):

                        bb=BoundBox(C)
                        bb.x = Bbox_pred[row, col, box_loop, 0]
                        bb.y = Bbox_pred[row, col, box_loop, 1]
                        bb.w = Bbox_pred[row, col, box_loop, 2]
                        bb.h = Bbox_pred[row, col, box_loop, 3]
                        bb.box = box_loop
                        bb.lab = labels[class_loop]
                        bb.probs = tempc
                        bb.prob_obj = Conf_pred[row, col, box_loop]
                        bb.prob_class = Classes[row, col, box_loop, class_loop]
                        boxes.append(bb)

                        
    return boxes#NMS(self, np.ascontiguousarray(probs).reshape(H*W*B,C), np.ascontiguousarray(_Bbox_pred).reshape(H*W*B,5))#boxes#

def findboxes(self, net_out):
    
    boxes = []
    if self.nms:
        boxes = box_constructor(self, net_out)
    else:
        boxes = box_constructor_sin_nms(self, net_out)
    
    return boxes

def process_box(self, b, h, w):
    max_prob = b.probs
    label = b.lab
    if max_prob > self.threshold:
        left  = int ((b.x - b.w/2.) * w)
        right = int ((b.x + b.w/2.) * w)
        top   = int ((b.y - b.h/2.) * h)
        bot   = int ((b.y + b.h/2.) * h)
        if left  < 0    :  left = 0
        if right > w - 1: right = w - 1
        if top   < 0    :   top = 0
        if bot   > h - 1:   bot = h - 1
        mess = '{}'.format(label)
        return (left, right, top, bot, mess, max_prob)
    return None


def postprocess(self, net_out, im, h, w):

    labels = self.labels
    colors = self.colors

    boxes = findboxes(self, net_out)
    
    imgcv = im.astype('uint8')
    resultsForJSON = []
    for b in boxes:
        
        """ Esta funcion auxiliar process_box se encarga de procesar las cajas """
        boxResults = process_box(self, b, h, w)
        if boxResults is None:
            continue
        
        left, right, top, bot, mess, confidence = boxResults
        resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
            
        
        cv2.rectangle(imgcv,(left, top), (right, bot),colors[labels.index(mess)], 2)
            
        confi = confidence*100

        if self.ver_probs:
            if top - 16 > 0:
                cv2.rectangle(imgcv,(left-1, top - 16), (left + (len(mess)+9)*5*2-1, top),colors[labels.index(mess)], -1)
                cv2.putText(imgcv,mess + ' -> ' + "%.2f" % confi  + '%' ,(left, top - 4), font, 0.45,(0,0,0),0,cv2.LINE_AA)
            else:
                cv2.rectangle(imgcv,(left-1, top), (left + (len(mess)+9)*5*2-1, top+16),colors[labels.index(mess)], -1)
                cv2.putText(imgcv,mess + ' -> ' + "%.2f" % confi  + '%' ,(left, top + 12), font, 0.45,(0,0,0),0,cv2.LINE_AA)
        else:
            if top - 16 > 0:
                cv2.rectangle(imgcv,(left-1, top - 16), (left + len(mess)*5*2-1, top),colors[labels.index(mess)], -1)
                cv2.putText(imgcv,mess,(left, top - 4), font, 0.45,(0,0,0),0,cv2.LINE_AA)
            else:
                cv2.rectangle(imgcv,(left-1, top), (left + len(mess)*5*2-1, top+16),colors[labels.index(mess)], -1)
                cv2.putText(imgcv,mess,(left, top + 12), font, 0.45,(0,0,0),0,cv2.LINE_AA)

    return resultsForJSON, imgcv


""" ===== ================ ===== """ """ ===== ================ ===== """ """ ===== ================ ===== """ """ ===== ================ ===== """ """ ===== ================ ===== """ """ ===== ================ ===== """
""" ===== FIN DEL PROGRAMA ===== """ """ ===== FIN DEL PROGRAMA ===== """ """ ===== FIN DEL PROGRAMA ===== """ """ ===== FIN DEL PROGRAMA ===== """ """ ===== FIN DEL PROGRAMA ===== """ """ ===== FIN DEL PROGRAMA ===== """
""" ===== ================ ===== """ """ ===== ================ ===== """ """ ===== ================ ===== """ """ ===== ================ ===== """ """ ===== ================ ===== """ """ ===== ================ ===== """
