
import numpy as np
import cv2

import os
import sys

""" ===================================
Subprograma auxiliar para eliminar los
elementos que haya en una ruta dada
=================================== """

def eliminar_elementos(ruta):
    lista_elementos = []
    for ruta, _, ficheros in os.walk(ruta):
        for nombre_fichero in ficheros:
            rut_comp = os.path.join(ruta, nombre_fichero)
            lista_elementos.append(rut_comp)
            
    for eliminando in lista_elementos:
        os.remove(eliminando)

""" =============================================================================================================================================================

Colocamos aquí la lista de parámetros que no varian a lo
largo de todo el programa y que todos los subprogramas
podran utilizar.

    *** Lista ***

0 - ent_numb_max -> Numero de veces que ejecutaremos el entrenamiento a lo sumo.
1 - paso_maximo -> Es el número de pasos que un lote puede entrenar antes de ser sustituido por otro lote. 
2 - perdida_minima -> Es la cantidad mínima que la función pérdida (f_p) puede tener. Si f_p < perdida_minima entonces cambiamos de lote.

3 - dim -> Es la dimensión de las imágenes con la que estamos tratando. Es difícil que este parámetro cambie.

4 - batch_size -> Tamaño que cada lote va a tener. Si tenemos una base de datos de tamaño n y batch_size = b_s entonces numero_de_lotes = [n/b_s].
5 - batch_size_test -> Tamaño del lote para el test.

6 - cada_pasito -> Variable entera que establece cada cuantos pasos (step) mostraremos el resultado de f_p, guardaremos datos y podremos optar a ver img_finales
7 - quiero_ver -> Variable de control booleana. Permite decirnos si queremos ver o no el resultado de los entrenamientos cada_pasito.

8 - learning_ratio -> Redio de aprendizaje. Es el número que regula el cambio de los pesos y sesgos de las capas convolucionales.

9 - threshold -> Nivel de precisión que ha de tener una predicción para crear una caja.

10 - labels -> Nombre de las posibles etiquetas.
11 - anchors -> Valores predeterminados para predicciones de cajas.

12 - H, W, S, C, B -> salida.shape[1], salida.shape[2], salida.shape[1 (o 2 que son iguales)], numero de clases, numero de cajas de predicción.
13 - sqrt -> Vete a saber tu para que sirve esto.

14 - n_final_layers -> Numero final de capas que ha de tener la salida.

============================================================================================================================================================= """

def prog_change_datos(ent_numb_max,paso_maximo,precision_min,dim_fil,dim_col,batch_size,batch_size_test,cada_pasito,quiero_ver,salvando,
                      preprocesamiento,learning_ratio,threshold,labels,anchors,H, W, C, B):
    
    print('')
    print('                  ===== PROGRAMA DE ENTRENAMIENTO =====')
    print('                  ===== ------------------------- =====')
    print('')
    print('                  ---> Red neuronal basada en YOLO <---')
    print('')
    print('                  *** PARAMETROS DE LA RED NEURONAL ***')
    print('                  ===== ------------------------- =====')
    print('')
    print('     1 - Numero maximo de entrenamientos a ejecutar ----------> ', ent_numb_max)
    print('     2 - Paso maximo -----------------------------------------> ', paso_maximo)
    print('     3 - Precision minima ------------------------------------> ', precision_min)
    print('')
    print('     4 - Dimension de las imagenes ---------------------------> ', dim_fil, ' - ', dim_col)
    print('')
    print('     5 - Tamaño del lote de entrenamiento --------------------> ', batch_size)
    print('     6 - Tamaño del lote de testeo ---------------------------> ', batch_size_test)
    print('')
    print('     7 - Numero de pasos para mostrar perdida y guardar ------> ', cada_pasito)
    print('     8 - Guardar imagenes de salida en "imagenes_devueltas" --> ', quiero_ver)
    print('     9 - Guardar entrenamiento -------------------------------> ', salvando)
    print('')
    print('     10 - Preprocesamiento -----------------------------------> ', preprocesamiento)
    print('     11 - Radio de aprendizaje -------------------------------> ', learning_ratio)
    print('')
    print('     =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*==*=*=*=*=*=*=*=*=*=*=*=*=*=*=')
    print('')
    print('     12 - Limite de precision aceptable ----------------------> ', threshold)
    print('     13 - Etiquetas ------------------------------------------> ', labels)
    print('     14 - Anclas (rectangulos predeterminados) ---------------> ', anchors)
    print('')
    print('     15 - H, W, S, C, B, sqrt --------------------------------> ', H, W, C, B)
    print('')
    scarlet = input('Estos son los parametros de la red. Está de acuerdo con ellos (s/n): ')
    l_p_scarlet = ['s', 'S', 'Y', 'y', 'si', 'Si', 'SI', 'sI', 'Yes', 'yes', 'YES', 'n', 'N', 'No', 'NO']

    while scarlet not in l_p_scarlet:

        print('')
        print('Introduzca correctamente la respuesta.')
        scarlet = input('Estos son los parametros de la red. Está de acuerdo con ellos (s/n): ')

    if scarlet in ['n', 'N', 'No', 'NO']:

        print('')
        print(' ATENCION. EL CAMBIO DE VALORES DE LA RED TRAE CONSECUENCIAS QUE ALTERARAN EL RESULTADO DE LA SALIDA')
        print('')

        de_acuerdo = 5
        while de_acuerdo not in ['s', 'S', 'Y', 'y', 'si', 'Si', 'SI', 'sI', 'Yes', 'yes', 'YES']:

            johansson = input('Introduzca el nombre del parametro de la lista que desea cambiar: ')
            lista_de_parametros = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14', '15']

            while johansson not in lista_de_parametros:

                print('')
                print('Introduce bien las cosas. No es tan dificil. Un numero del 1 al 15.')
                johansson = input('Introduzca el nombre del parametro de la lista que desea cambiar: ')

            if johansson == '1':

                print('')
                print('Vas a cambiar el numero de entrenamiento maximos. Su valor actual es: ', ent_numb_max)
                nuevo_valor = input('Introduce un nuevo valor: ')

                try:
                    ent_numb_max = int(nuevo_valor)
                except:
                    print('')
                    print('La has cagado. Adios.')
                    sys.exit()

            elif johansson == '2':

                print('')
                print('Vas a cambiar el numero de pasos maximos. Su valor actual es: ', paso_maximo)
                nuevo_valor = input('Introduce un nuevo valor: ')

                try:
                    paso_maximo = int(nuevo_valor)
                except:
                    print('')
                    print('La has cagado. Adios.')
                    sys.exit()

            elif johansson == '3':

                print('')
                print('Vas a cambiar la precision minima. Su valor actual es: ', precision_min)
                nuevo_valor = input('Introduce un nuevo valor: ')

                try:
                    precision_min = float(nuevo_valor)
                except:
                    print('')
                    print('La has cagado. Adios.')
                    sys.exit()

            elif johansson == '4':

                print('')
                print('Vas a cambiar la imension de las imagenes (NO RECOMENDADO). Su valor actual es: ', dim_fil, ' - ', dim_col)
                nuevo_valor_fil = input('Introduce un nuevo valor de dim_fil: ')
                nuevo_valor_col = input('Introduce un nuevo valor de dim_col: ')

                try:
                    dim_fil, dim_col = int(nuevo_valor_fil), int(nuevo_valor_col)
                except:
                    print('')
                    print('La has cagado. Adios.')
                    sys.exit()

            elif johansson == '5':

                print('')
                print('Vas a cambiar el tamano del lote. Su valor actual es: ', batch_size)
                nuevo_valor = input('Introduce un nuevo valor: ')

                try:
                    batch_size = int(nuevo_valor)
                except:
                    print('')
                    print('La has cagado. Adios.')
                    sys.exit()

            elif johansson == '6':

                print('')
                print('Vas a cambiar el numero de elementos para testear. Su valor actual es: ', batch_size_test)
                nuevo_valor = input('Introduce un nuevo valor: ')

                try:
                    batch_size_test = int(nuevo_valor)
                except:
                    print('')
                    print('La has cagado. Adios.')
                    sys.exit()

            elif johansson == '7':

                print('')
                print('Vas a cambiar el numero de pasos para mostrar perdida y guardar parametro e imagenes. Su valor actual es: ', cada_pasito)
                nuevo_valor = input('Introduce un nuevo valor: ')

                try:
                    cada_pasito = int(nuevo_valor)
                except:
                    print('')
                    print('La has cagado. Adios.')
                    sys.exit()

            elif johansson == '8':

                print('')
                print('Vas a cambiar la opcion para ver las imagenes en carpeta. Su valor actual es: ', quiero_ver)
                nuevo_valor = input('Introduce un nuevo valor: ')

                try:
                    quiero_ver = nuevo_valor
                except:
                    print('')
                    print('La has cagado. Adios.')
                    sys.exit()

            elif johansson == '9':

                print('')
                print('Vas a cambiar la opcion para salvar. Su valor actual es: ', salvando)
                nuevo_valor = input('Introduce un nuevo valor: ')

                try:
                    salvando = nuevo_valor
                except:
                    print('')
                    print('La has cagado. Adios.')
                    sys.exit()

            elif johansson == '10':

                print('')
                print('Vas a cambiar el preprocesamiento de imagenes. Su valor actual es: ', preprocesamiento)
                nuevo_valor = input('Introduce un nuevo valor: ')

                try:
                    preprocesamiento = nuevo_valor
                except:
                    print('')
                    print('La has cagado. Adios.')
                    sys.exit()

            elif johansson == '11':

                print('')
                print('Vas a cambiar el radio de aprendizaje. Su valor actual es: ', learning_ratio)
                nuevo_valor = input('Introduce un nuevo valor: ')

                try:
                    learning_ratio = float(nuevo_valor)
                except:
                    print('')
                    print('La has cagado. Adios.')
                    sys.exit()

            elif johansson == '12':

                print('')
                print('Vas a cambiar el limite de precision. Su valor actual es: ', threshold)
                nuevo_valor = input('Introduce un nuevo valor: ')

                try:
                    threshold = float(nuevo_valor)
                except:
                    print('')
                    print('La has cagado. Adios.')
                    sys.exit()

            elif johansson == '13':

                print('')
                print('Vas a cambiar elas etiquetas. Su valor actual es: ', labels)
                nuevo_valor = input('Introduce un nuevo valor (ej: a b c): ')

                try:
                    labels = nuevo_valor.split(' ')
                except:
                    print('')
                    print('La has cagado. Adios.')
                    sys.exit()

            elif johansson == '14':

                print('')
                print('Vas a cambiar las anclas (NO RECOMENDADO). Su valor actual es: ', anchors)
                nuevo_valor = input('Introduce un nuevo valor (ej: 1 2 3 4 5): ')

                try:
                    anchors_2 = nuevo_valor.split(', ')
                    anchors = []
                    for avc in anchors_2:
                        anchors.append(float(avc))
                        
                except:
                    print('')
                    print('La has cagado. Adios.')
                    sys.exit()

            elif johansson == '15':

                print('')
                print('Vas a cambiar alguno de los valores H, W, C, B.')
                megan = input('Cual de ellos: ')

                if megan in ['H', 'W', 'C', 'B']:

                    if megan == 'H':

                        print('')
                        try:
                            H = int(input('Introduce un nuevo valor: '))
                        except:
                            print('')
                            print('La has cagado. Adios.')
                            sys.exit()

                    if megan == 'W':

                        print('')
                        try:
                            W = int(input('Introduce un nuevo valor: '))
                        except:
                            print('')
                            print('La has cagado. Adios.')
                            sys.exit()

                    if megan == 'C':

                        print('')
                        try:
                            C == int(input('Introduce un nuevo valor: '))
                        except:
                            print('')
                            print('La has cagado. Adios.')
                            sys.exit()

                    if megan == 'B':

                        print('')
                        try:
                            B = int(input('Introduce un nuevo valor: '))
                        except:
                            print('')
                            print('La has cagado. Adios.')
                            sys.exit()

                else:
                    print('')
                    print('La has cagado. Adios.')
                    sys.exit()
                    
            print('')
            print('                  ===== PROGRAMA DE ENTRENAMIENTO =====')
            print('                  ===== ------------------------- =====')
            print('')
            print('                  ---> Red neuronal basada en YOLO <---')
            print('')
            print('                  *** PARAMETROS DE LA RED NEURONAL ***')
            print('                  ===== ------------------------- =====')
            print('')
            print('     1 - Numero maximo de entrenamientos a ejecutar ----------> ', ent_numb_max)
            print('     2 - Paso maximo -----------------------------------------> ', paso_maximo)
            print('     3 - Precision minima ------------------------------------> ', precision_min)
            print('')
            print('     4 - Dimension de las imagenes ---------------------------> ', dim_fil, ' - ', dim_col)
            print('')
            print('     5 - Tamaño del lote de entrenamiento --------------------> ', batch_size)
            print('     6 - Tamaño del lote de testeo ---------------------------> ', batch_size_test)
            print('')
            print('     7 - Numero de pasos para mostrar perdida y guardar ------> ', cada_pasito)
            print('     8 - Guardar imagenes de salida en "imagenes_devueltas" --> ', quiero_ver)
            print('     9 - Guardar entrenamiento -------------------------------> ', salvando)
            print('')
            print('     10 - Preprocesamiento -----------------------------------> ', preprocesamiento)
            print('     11 - Radio de aprendizaje -------------------------------> ', learning_ratio)
            print('')
            print('     =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*==*=*=*=*=*=*=*=*=*=*=*=*=*=*=')
            print('')
            print('     12 - Limite de precision aceptable ----------------------> ', threshold)
            print('     13 - Etiquetas ------------------------------------------> ')#, labels)
            print('     14 - Anclas (rectangulos predeterminados) ---------------> ', anchors)
            print('')
            print('     15 - H, W, S, C, B, sqrt --------------------------------> ', H, W, C, B)
            print('')
            de_acuerdo = input('Estos son los parametros de la red. Está de acuerdo con ellos (s/n): ')

        print('')
        print('Perfecto, alla vamooooos.')
        
    else:

        print('')
        print('Perfecto, continuemos.')

    return ent_numb_max,paso_maximo,precision_min,dim_fil,dim_col,batch_size,batch_size_test,cada_pasito,quiero_ver,salvando,preprocesamiento,learning_ratio,threshold,labels,anchors,H, W, C, B

def desordenar(nombrecitos_bonitos):

    longi = len(nombrecitos_bonitos)
    lista_aleatoria = np.random.randint(0, longi, (longi))

    if longi != 1:
        for lana in range(int(longi/2)):

            num1 = lista_aleatoria[2*lana]
            num2 = lista_aleatoria[2*lana + 1]
            
            aux = nombrecitos_bonitos[num1]

            nombrecitos_bonitos[num1] = nombrecitos_bonitos[num2]
            nombrecitos_bonitos[num2] = aux

    return nombrecitos_bonitos

def desordenar_todo(nombrecitos_bonitos):

    longi = len(nombrecitos_bonitos)
    lista_aleatoria = np.random.randint(0, longi, (longi))

    if longi != 1:
        for lana in range(int(longi/2)):

            num1 = lista_aleatoria[2*lana]
            num2 = lista_aleatoria[2*lana + 1]
            
            aux = nombrecitos_bonitos[num1]

            nombrecitos_bonitos[num1] = nombrecitos_bonitos[num2]
            nombrecitos_bonitos[num2] = aux

    return nombrecitos_bonitos

""" ===================================
Este programa carga el lote. Nosotros
le damos los nombres de los archivos
.txt y el nos devuelve la imagen una
vez realizadas algunas transformaciones
aparte de los array probabilidad,
coordenada, area...
=================================== """
