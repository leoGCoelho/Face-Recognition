import os
import numpy as np
import cv2
from PIL import Image
import pickle


#========= Folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

#========= Haar Cascade and Recogniter
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

#========= Global variables
current_id = 0
label_ids = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk(image_dir):
    for file in files:

#========= Search for any .png or .jpg files in /images/
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            #print(label, path)

#========= Update labels vector
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label] #label is based on folder name
            #print(label_ids)

#========= Adjust images (grayscale)
            pil_image = Image.open(path).convert("L")   #grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image, "uint8")
            #print(image_array)

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

#========= Identify ROI (region of interest) and make element tag and label
            for (x, y, w, h) in faces:
                rdi = image_array[y:y+h, x:x+w]      #[Y-start:Y-end, X-start:X-end]
                x_train.append(rdi)
                y_labels.append(id_)
                

#print(y_labels)
#print(x_train)
#print(label_ids)

#========= Generate bin archieve with labels data
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

#========= Generate YML with all info data based on image pixel and label Id
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")

print("Trainning Complete!")