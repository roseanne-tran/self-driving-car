from keras.models import load_model
import cv2
import numpy as np
import csv


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalizeHist(img):
    img = cv2.equalizeHist(img)
    return img

def preprocess_image(img):
    img = grayscale(img)
    img = equalizeHist(img)
    img = img/255
    return img
    

def process_labels(fileName):
    with open(fileName) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            classes.append(row[0])
            description.append(row[1])
        
        


#process labels
classes = []
description = []

process_labels('labels.csv')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

#load model
model = load_model('model.h5')

#print('class size: ', len(classes))
#print('description size: ', len(description))


#
#for c in classes:
#    print(c)
#
#for des in description:
#    print(des)
    
while True:

   # READ IMAGE
   _, frame = cap.read()

   # PROCESS IMAGE
   processed_image = np.asarray(frame)
   processed_image = cv2.resize(processed_image, (32, 32))
   processed_image = preprocess_image(processed_image)
#   cv2.imshow('RESULT', processed_image)
   processed_image = processed_image.reshape(1, 32, 32, 1)
   cv2.putText(frame, "CLASS: " , (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
   cv2.putText(frame, "PROBABILITY: ", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
   
   # PREDICT IMAGE
   prediction = model.predict(processed_image)
   predict_class = model.predict_classes(processed_image)
#   print('predicted class: ', predict_class)
#   print('predicted class len: ', len(predict_class))

   probability = np.amax(prediction)
   
   if probability > 0.8:
#       print('in here')
       cv2.putText(frame,str(predict_class)+" "+str(description[predict_class[0]+1]), (120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
       cv2.putText(frame, str(round(probability*100,2) )+"%", (180, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
   cv2.imshow("RESULT", frame)

   if cv2.waitKey(1) and 0xFF == ord('q'):
       break

