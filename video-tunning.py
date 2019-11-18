"""
DIMENSIONES INICIALES DE VIDEO: 480x360
"""
import cv2
import numpy as np
REZISE_INTERPOLATION_METHODS = {
    'nearest_neighbour': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'pixel_area_relation': cv2.INTER_AREA,
    'bicubic': cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4
}

# bajado de resolucion usando piramide gaussiana
def bajar_resolucion(video_name):
    cap = cv2.VideoCapture(video_name)

    # le bajo la resolucion a su cuarta parte
    nuevo_ancho = int(cap.get(3) / 2)
    nuevo_alto = int(cap.get(4) / 2)
    # cap.set(3, nuevo_ancho)
    # cap.set(4, nuevo_alto)
    nuevas_dimensiones = (nuevo_ancho, nuevo_alto)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('resolucion-bajada.avi', fourcc, 30, nuevas_dimensiones)
    while True:
        ret, frame = cap.read()
        
        if ret == True:
            # aplico piramide gaussiana para bajar la resolucion a la cuarta parte
            bajada = cv2.pyrDown(frame)
            # print(frame.shape, bajada.shape)
            out.write(bajada)
        else:
            print('resolucion-bajada OK')
            break
    cap.release()
    out.release()

def interpolar_video(video_name, interpolation_name, interpolation_method):
    cap = cv2.VideoCapture(video_name)
    nuevas_dimensiones = (4096, 2160)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(interpolation_name+'.avi', fourcc, 30, nuevas_dimensiones)
    while True:
        ret, frame = cap.read()
        if ret == True:
            # aplico tecnicas convencionales de interpolacion
            b = cv2.resize(frame, nuevas_dimensiones, fx = 0, fy = 0, interpolation = interpolation_method)
            out.write(b)
        else:
            print(interpolation_name, 'OK')
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
# bajo la resolucion al video
# y se genera un nuevo video con el nombre resolucion-bajada.avi
bajar_resolucion('video.mp4')

# con el video con resolucion bajada trabajamos para subirla mediante tecnicas de interpolacion
for interpolation_name, interpolation_method in REZISE_INTERPOLATION_METHODS.items():
    interpolar_video('resolucion-bajada.avi', interpolation_name, interpolation_method)