import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np


facedetect = cv2.CascadeClassifier('C:/Users/Sebi/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

#cream captura video de 640x480 pixeli
cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font=cv2.FONT_HERSHEY_COMPLEX


model = load_model('C:/Users/Sebi/proiect_ps/Face Recognition System/keras_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#functie pentru determinarea numelui in functie de index ul clasei din care face parte 
def get_className(classNo):
	if classNo==0:
		return "Sebi Deaconu"
	elif classNo==1:
		return "Maddison Beer"
	elif classNo==2:
		return "Tonny Stark"
	elif classNo==3:
		return "Jenna Ortega"
	
		

while True:
	sucess, imgOrignal=cap.read()
	faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
	for x,y,w,h in faces:
		crop_img=imgOrignal[y:y+h,x:x+h]
		#facem resize la imagine pentru a se potrivi cu dimensiunea de intrare a retelei neuronale 
		img=cv2.resize(crop_img, (224,224))
		img=img.reshape(1, 224, 224, 3)
		image_array = np.asarray(img)
		#normalizam imaginea pentru a avea rezultate cat mai optime
		normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
		prediction=model.predict(normalized_image_array)
		classIndex=np.argmax(prediction)
		probabilityValue=prediction[0][classIndex]
		if classIndex==0:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
			cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
		elif classIndex==1:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
			cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
		elif classIndex==2:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
			cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
		elif classIndex==3:
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
			cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)		
		cv2.putText(imgOrignal,str(round(probabilityValue*100, 2))+"%" ,(180, 75), font, 0.75, (255,0,0),2, cv2.LINE_AA) #afisarea procentajului
	cv2.imshow("Result",imgOrignal)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break


cap.release()
cv2.destroyAllWindows()
















