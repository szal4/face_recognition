import os
import cv2
from PIL import Image

face_cascade = cv2.CascadeClassifier(r"C:\Users\dp\AppData\Local\Programs\Python\Python38-32\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
rec = cv2.face.LBPHFaceRecognizer_create() #local binary pattern histogram algorithum
rec.read('trained_data.yml')
id=0
path = os.getcwd()
data_path = os.path.join(path,'images')

ddd = {}
count=1
for files,roots,dirs in os.walk(data_path):
    for root in roots:
        ddd[''+root+'']=count
        count+=1

inv_ddd = {v:k for k,v in ddd.items()}
cam = cv2.VideoCapture(0)
while(True):
    _,image = cam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.95, 3)
    for x,y,w,h in faces:
        image = cv2.rectangle(image, (x,y), (x+w,y+h),(0,255,0),3)
        roi_gray = gray[y:y+h, x:x+w]
        resized = cv2.resize(roi_gray,(250,250))
        ide,conf = rec.predict(resized)
        per = int(conf)
        cv2.putText(image,''+str(per)+'%',(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
        cv2.putText(image,''+str(inv_ddd.get(int(ide)))+'',(x,y+h),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
    cv2.imshow("faceeee",image)
    if cv2.waitKey(1)==27:
        break
cam.release()
cv2.destroyAllWindows()
    
