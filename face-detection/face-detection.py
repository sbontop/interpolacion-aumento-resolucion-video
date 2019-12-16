import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.
# cap = cv2.VideoCapture(0)
# To use a video file as input
cap = cv2.VideoCapture('rostroV.mp4')
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
