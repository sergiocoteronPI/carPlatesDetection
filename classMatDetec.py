
import numpy as np

tkinterOK = True
try:
    import tkinter
    from tkinter import *
    import tkinter.filedialog
    from tkinter.filedialog import askopenfilename
except:
    tkinterOK = False
    print("Error: no ha sido posible importar la libreria tkinter o tkinter.filedialog")
    print("Es posible que surjan problemas a lo largo de la ejecución del código.")

class claseParametrosDeteccionMatriculas:

    def __init__(self, threshold, batch_size, dim_fil, dim_col, H, W, B, learning_ratio, nms, ver_probs, rpe, rpi, h5):
        
        self.threshold = threshold
        self.batch_size = batch_size

        self.dim_fil = dim_fil
        self.dim_col = dim_col

        self.labels = ['matricula']
        
        #self.anchors = [0.57273,0.677385, 1.87446,2.06253, 3.33843,5.47434, 7.77052,7.16828,  16.62,10.5][0:2*B]
        self.anchors = [1,1, 1,1, 1,1, 1,1, 1,1, 1,1][0:2*B]

        self.H = H
        self.W = W
        self.C = len(self.labels)
        self.B = B
        self.HW = H*W

        self.colors = np.random.randint(0,255 ,(self.C,3)).tolist()
        self.colors[0] = [255,0,255]
        self.learning_ratio = learning_ratio

        self.nms = nms
        self.ver_probs = ver_probs

        self.clases_visibles = [self.labels.index(v) for v in self.labels]

        self.rpe = rpe
        self.rpi = rpi

        self.h5 = h5

    def cambiarParametro(self):

        queHacer = "noSalir"
        
        while queHacer not in ["salir", "exit", "", " "]:

            parametrosName = ["threshold", "batch_size", "dim_fil", "dim_col", "H", "W", "B", "learning_ratio", "nms", "ver_probs", "rpe", "rpi", "h5"]
            parametros = [self.threshold, self.batch_size, self.dim_fil, self.dim_col, self.H, self.W, self.B, self.learning_ratio, self.nms, self.ver_probs, self.rpe, self.rpi, self.h5]

            print("")
            print(" Estos son los parámetros actuales. Seleccione el numero del que quiera modificar ")
            print(" ================================================================================ ")
            print("")
            for i in range(1, len(parametrosName)+1):
                print(str(i) + " - " + parametrosName[i-1] + ":", parametros[i-1])
            print("")
            print(" ================================================================================ ")
            print("")
            queHacer = input(" numero/salir/exit: ")
            print("")
            print(" ================================================================================ ")

            if queHacer == str(11):
                self.setRutaParaEtiquetas()

            if queHacer == str(12):
                self.setRutaParaImagenes()

    def setRutaParaImagenes(self):

        if tkinterOK:
            self.rpi = tkinter.filedialog.askdirectory(initialdir = "./",title = "Elije un directorio")
            self.rpi +=  "/"

    def setRutaParaEtiquetas(self):

        if tkinterOK:
            self.rpe = tkinter.filedialog.askdirectory(initialdir = "./",title = "Elije un directorio")
            self.rpe +=  "/"


classMatDetec = claseParametrosDeteccionMatriculas(threshold = 0.5,
                                                   batch_size = 5,
                                                   dim_fil = 480, dim_col = 480,
                                                   H = 13, W = 13, B = 3,
                                                   learning_ratio = 1e-3,
                                                   nms = True,
                                                   ver_probs = True,
                                                   rpe = "C:/Users/sergio.coteron/Desktop/proyectosPython/proyectoMatriculas/carPlatesDetection/baseDeDatos/label/",
                                                   rpi = "C:/Users/sergio.coteron/Desktop/proyectosPython/proyectoMatriculas/carPlatesDetection/baseDeDatos/images/",
                                                   h5 = 'mark1_matdetec.h5')