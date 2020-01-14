import xml.etree.ElementTree as ET

import os
import numpy as np
import cv2


listaXml = []

for ruta, _, archivos in os.walk("XML/"):
    for nombreArchivo in archivos:
        rutaCompleta = os.path.join(ruta, nombreArchivo)

        if(rutaCompleta.endswith("xml")):
            listaXml.append(rutaCompleta)

for name in listaXml:        

    tree = ET.parse(name)
    root = tree.getroot()

    ruta = root.find("./folder").text + '/' + root.find("./filename").text

    lineas = []
    for elem in root.findall("./object"):

        left = elem.find("./bndbox/xmin").text
        top = elem.find("./bndbox/ymin").text
        right = elem.find("./bndbox/xmax").text
        bot = elem.find("./bndbox/ymax").text

        lineas.append("falsosNegativos/" + root.find("./filename").text + ',' + str(left) + ',' + str(top) + ',' + str(right) + ',' + str(bot) + '\n')

    try:
        img = cv2.imread(ruta)
    except:
        print("No se ha podido abrir la imagen: " + ruta)

    if lineas != []:
        nombreTxtFalso = "falsosNegativosLabel/" + os.path.basename(root.find("./filename").text).split(".")[0] + ".txt"
        with open(nombreTxtFalso, 'w') as f:
            for line in lineas:
                f.write(line)

        try:
            cv2.imwrite("falsosNegativos/" + root.find("./filename").text, img)
        except:
            print("No se ha podido guardar la imagen en: " + "falsosNegativos/" + root.find("./filename").text, img)

    else:

        os.remove(ruta)
        os.remove(name)

    #os.remove(name)