from imutils import face_utils
import dlib
import cv2
 
# vamos a inicializar y codificar un detector de caras (HOG) 
#y después de detectar los puntos de referencia en esta cara detectada

# p = nuestro directorio modelo pre-entrenado, está en el mismo directorio del script.
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture('rostroV.mp4')
cont=0
rostrosEnImagen=0
numTotal=0
 
while True:
    _, image = cap.read()
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Detectar rostros en el video
    rects = detector(gray, 2)
    if len(rects) == 0:
        print("No faces found")
    else:
        print("Number of faces detected: ",len(rects))
        rostrosEnImagen=len(rects)
        cont=cont+1
        if cont == 1:
            numTotal=numTotal+rostrosEnImagen
        if cont == 30:
            cont=0

    # Para cada cara detectada, encuentre el punto de referencia.
    for (i, rect) in enumerate(rects):
        # Haga la predicción y transfórmela en una matriz numpy
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        #convertir el rectángulo de dlib en un cuadro delimitador de estilo OpenCV
        (x,y,w,h)=face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x,y), (x+w,y+h),(255,0,0),2) #azul
        
        # Dibuje en nuestra imagen, todos los puntos de coordenadas de búsqueda (x, y)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1) #verde
        
        cv2.putText(image, "Number of faces detected: " + str(len(rects)), (1, image.shape[0] - 5), cv2.FONT_ITALIC, 0.45,  (0, 0, 255), lineType=cv2.LINE_AA)
        cv2.putText(image, "Total number of faces: " + str(numTotal), (570, image.shape[0] - 5), cv2.FONT_ITALIC, 0.45,  (0, 0, 255), lineType=cv2.LINE_AA)

    
    # Muestra la imagen
    cv2.imshow("Deteccion rostros", image)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27: #Tecla esc
        break

cv2.destroyAllWindows()
cap.release()