from PIL import Image
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
import cv2
from matplotlib.image import imread
import argparse
import numpy as np

#Agrupamos todas imagenes en una lista 
import os 
Listaimagenes=[]

def getFiles(path):
    for file in os.listdir(path):
        if file.endswith(".pgm"):
            Listaimagenes.append(os.path.join(path, file))

filesPath = '/content/drive/MyDrive/Segmentacion_Angio/Angiograms'

getFiles(filesPath)
print(Listaimagenes)

def rdp(points, epsilon):
    """
    Aplica el algoritmo de Ramer-Douglas-Peucker a una serie de puntos.

    :param points: lista de puntos representando la curva a simplificar
    :param epsilon: distancia máxima permitida entre el punto original y la aproximación
    :return: lista de puntos simplificados
    """
    # Encontrar el punto con la máxima distancia
    d_max = 0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_to_line_distance(points[i], points[0], points[-1])
        if d > d_max:
            index = i
            d_max = d

    # Si la distancia máxima es mayor que el epsilon dado, dividir la curva en dos y simplificar cada mitad
    if d_max > epsilon:
        left_points = points[:index+1]
        right_points = points[index:]
        left_simplified = rdp(left_points, epsilon)
        right_simplified = rdp(right_points, epsilon)
        return left_simplified[:-1] + right_simplified
    else:
        return [points[0], points[-1]]

def point_to_line_distance(point, line_start, line_end):
    """
    Calcula la distancia entre un punto y una línea definida por dos puntos.

    :param point: punto a calcular la distancia
    :param line_start: primer punto que define la línea
    :param line_end: segundo punto que define la línea
    :return: distancia entre el punto y la línea
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    denominator = ((y2-y1)**2 + (x2-x1)**2)**0.5
    return numerator / denominator


def escala_de_grises(str):
    #Abrimos la Imagen
    im = Image.open(str)
    im.show()
    #Obtenemos sus dimensiones
    x = im.size[0]
    y = im.size[1]
    #Creamos una nueva imagen con las dimensiones de la imagen anterior
    im2 = Image.new('RGB', (x, y))
    i = 0
    while i < x:
        j = 0
        while j < y:
            #Obtenemos el valor RGB de cada pixel
            r, g, b = im.getpixel((i,j))
            #Obtenemos su equivalente en la escala de gris
            p = (r * 0.3 + g * 0.59 + b * 0.11)
            #Ese valor lo convertimos a entero
            gris = int(p)
            pixel = tuple([gris, gris, gris])
            #En la nueva imagen en la posición i, j agregamos el nuevo color 
            im2.putpixel((i,j), pixel)
            j += 1
        i += 1
    #Guardamos la imagen
    im2.save(str)
    im2.show()

# Vamos a definir una nueva funcion para enseñar imagenes
def imshow(title="Image", image = None, size = 10):
  w, h = image.shape[0], image.shape[1]
  aspect_ratio = w/h
  plt.figure(figsize=(size*aspect_ratio,size))
  plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  plt.title(title)
  plt.show()

contador = 0
for i in Listaimagenes:
  contador = contador + 1
  imagen  = cv2.imread(i)
  imshow(str(contador),imagen)

for j in range(0,len(Listaimagenes)):
  #Usando el kernel Top-Hat 
  filter_size = (25,25)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,filter_size)
  #leyendo la imagen llamada 
  input_image = cv2.imread(Listaimagenes[j])
  aver = cv2.imread(Listaimagenes[j])
  input_image = cv2.cvtColor(input_image,cv2.COLOR_RGB2GRAY)

  #aplicando la operacion Top-Hat
  tophat_img = cv2.morphologyEx(input_image,cv2.MORPH_BLACKHAT,kernel)
  imshow("Tophat"+str(j),tophat_img)
  #Umbralizacion Otsu
  ret2,th2 = cv2.threshold(tophat_img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  #Umbralizar
  image = th2
  #Conectividad (componentes conectados)
  connectivity =  8
  #Aplicamos Componentes Conectados a la imagen umbralizada
  output = cv2.connectedComponentsWithStats(th2,connectivity,cv2.CV_32S)
  (numLabels,labels,stats,centroids) = output
  #inicializa la mascara de la salida para guardar todos los caracteres 
  #parsed from the license plate
    # initialize an output mask to store all characters parsed from
  # the license plate
  mask = np.zeros(th2.shape, dtype="uint8")
  print("Numero de etiquetas", numLabels)  # loop over the number of unique connected component labels, skipping
  # over the first label (as label zero is the background)

  for i in range(1, numLabels):
    # extract the connected component statistics for the current
    # label
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    # ensure the width, height, and area are all neither too small
    # nor too big
    keepWidth = w > 0 
    keepHeight = h > 30 
    keepArea = area > 60
    # ensure the connected component we are examining passes all
    # three tests
    if all((keepWidth, keepHeight, keepArea)):
      # construct a mask for the current connected component and
      # then take the bitwise OR with the mask
      print("[INFO] keeping connected component '{}'".format(i))
      componentMask = (labels == i).astype("uint8") * 255
      mask = cv2.bitwise_or(mask, componentMask)
    
  # show the original input image and the mask for the license plate
  # characters
  imshow("Imagen de " + str(j), th2)
  imshow("Filtrado de " + str(j), mask)
  cv2.imwrite('/content/drive/MyDrive/Segmentacion_Angio/imagenesdump/angio' + str(j+1)+"_F.pgm", mask)
  img = mask
  size = np.size(img)
  skel = np.zeros(img.shape,np.uint8)

  element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
  done = False
  #skeletizacion 
  while( not done):
      eroded = cv2.erode(img,element)
      temp = cv2.dilate(eroded,element)
      temp = cv2.subtract(img,temp)
      skel = cv2.bitwise_or(skel,temp)
      img = eroded.copy()

      zeros = size - cv2.countNonZero(img)
      if zeros==size:
          done = True
  # Can we make this in black and white? grayscale
  image_gray = np.zeros(skel.shape, np.uint8) 
  # Thickness - if positive. Negative thickness means that it is filled
  cv2.rectangle(image_gray, (64,64), (448,448), (255), -1)
  skel = cv2.bitwise_and(image_gray, skel)
  imshow("Esquelitizacion quitando el borde",skel)

  # Leaf directory 
  directory = "RGBParcheImagen" + str(j)
  # Parent Directories 
  parent_dir = "/content/drive/MyDrive/Segmentacion_Angio/imagenesdump"      
  # Path 
  path = os.path.join(parent_dir, directory) 
  # Create the directory 
  # Parche Imagen
  os.makedirs(path) 
  M = int(128/2) #
  N = int(128/2) #
  coordenadas = []
  for s in range(0,len(skel)):
    for t in range(0,len(skel)):
        if skel[s][t] == 255:
          print("Coordenada", s ,t)
          coordenadas.append((s,t))
          #guardar Lista con las coordenadas 
          #randolph douglas peucker seria aqui y luego guardar la imagen ya segmentada 
          #python append Point list[s,t] ((),())
          cv2.imwrite(directory + "/Parche_" + str(s-M) + "x" + str(t-N) + "_" + str(s+M) + "x" + str(t+N) + ".pgm", aver[s-M:s+M,t-N:t+N])

# Lista de coordenadas
coordenadas
# Aplicar el algoritmo RDP con un valor de epsilon de 1
simplified_coordinates = rdp(coordenadas, 5)

# Imprimir las coordenadas simplificadas
print(simplified_coordinates)

import matplotlib.pyplot as plt

# Lista de coordenadas
coordenadas

# Aplicar el algoritmo RDP con un valor de epsilon de 1
simplified_coordinates = rdp(coordenadas, 1)

# Separar las coordenadas en listas separadas de x e y
x_orig = [point[0] for point in coordenadas]
y_orig = [point[1] for point in coordenadas]
x_simplified = [point[0] for point in simplified_coordinates]
y_simplified = [point[1] for point in simplified_coordinates]

# Graficar la curva original y la curva simplificada
plt.plot(x_orig, y_orig, label='Original')
plt.plot(x_simplified, y_simplified, label='Simplificado')
plt.legend()
plt.show()
plt.savefig('imagen1.png',dpi=340)