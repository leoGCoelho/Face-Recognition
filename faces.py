import numpy as np
import cv2
import pickle

#========= HaarCascades
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

#========= Recognizer Attributes
recognizer.read("trainner.yml")
labels = {"persons_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

#========= Global Variables
cap = cv2.VideoCapture(0)

while(True):
    #========= Image Caption
    ret, frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayframe, scaleFactor=1.5, minNeighbors=5)

    for(x, y, w, h) in faces:
        #========= Coordinates and Region of Interest (regiao de interece)
        print(x, y, w, h)
        rdi_gray = grayframe[y:y+h, x:x+w]      #[Y-start:Y-end, X-start:X-end]
        rdi_color = frame[y:y+h, x:x+w]

        #========= Recognizer
        id_, conf = recognizer.predict(rdi_gray)
        if (conf >= 45): # and (conf <= 85):
            print(id_, labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        #========= Results Prints

            #====== Image Write
        g_img_item = "results/img-gray.png"
        c_img_item = "results/img-color.png"
        cv2.imwrite(g_img_item, rdi_gray)
        cv2.imwrite(c_img_item, rdi_color)

            #====== Image Captor
        color = (0, 255, 0)     # BGR
        stroke = 2      # expessura
        x_end = x + w
        y_end = y + h
        cv2.rectangle(frame, (x, y), (x_end, y_end), color, stroke)


    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cap.destroyAllWindows()