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

SUPER_RESOLUTION_VIDEOS = [
    'nearest_neighbour.avi',
    'bilinear.avi',
    'pixel_area_relation.avi',
    'bicubic.avi',
    'lanczos.avi'
]

def bajar_resolucion(video_name):
    """
    bajado de resolucion usando piramide gaussiana
    """
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
    """
    Interpolacion de videos a partir del video con resolucion bajada a su cuarta parte
    """
    cap = cv2.VideoCapture(video_name)
    nuevas_dimensiones = (int(cap.get(3) * 2), int(cap.get(4) * 2))
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

def contar_rostros_haar_cascade(video_name):
    """
    Reconocer y contar rostros de los videos con resolucion elevada usando la tecnica haar cascade
    """
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('face-detection/haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(video_name)
    cont=0
    rostrosEnImagen=0
    numTotal=0
    while True:
        # capture frame by frame

        # Read the frame
        _, img = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        # faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        print(type(faces), faces)
        if len(faces) == 0:
            print("No faces found")
        else:
            print("Number of faces detected: " + str(faces.shape[0]))
            rostrosEnImagen=faces.shape[0]
            cont=cont+1
            if cont == 1:
                numTotal=numTotal+rostrosEnImagen
            if cont == 30:
                cont=0

            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)

            cv2.rectangle(img, ((0, img.shape[0] - 25)),
                        (270, img.shape[0]), (255, 255, 255), -1)
            cv2.putText(img, "Number of faces detected: " + str(
                faces.shape[0]), (1, img.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.47,  (0, 0, 0), lineType=cv2.LINE_AA)

        # Display
        cv2.imshow('img', img)
            
        # Stop if q key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print("NUmero total de rostros: ",numTotal)
    # Release the VideoCapture object
    cap.release()
    
# bajo la resolucion al video
# y se genera un nuevo video con el nombre resolucion-bajada.avi

bajar_resolucion('video.mp4')

# con el video con resolucion bajada trabajamos para subirla mediante tecnicas de interpolacion

for interpolation_name, interpolation_method in REZISE_INTERPOLATION_METHODS.items():
    interpolar_video('resolucion-bajada.avi', interpolation_name, interpolation_method)

# leo los videos con resolucion elevada y cuento sus rostros
for video_name in SUPER_RESOLUTION_VIDEOS:
    # print(video_name)
    contar_rostros_haar_cascade(video_name)
