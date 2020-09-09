import os
import cv2
from PIL import Image
face_cascade = cv2.CascadeClassifier(r"C:\Users\dp\AppData\Local\Programs\Python\Python38-32\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

path = os.getcwd()
outpath = os.path.join(path,'images')   
cam = cv2.VideoCapture(0)
user_name = input('enter the name of user ')
user_name = user_name.replace(' ','_')
try:
    os.mkdir(os.path.join(outpath,''+user_name+''))
except:
    print('folder named '+user_name+' is already created')
    pass
os.chdir(os.path.join(outpath,''+user_name+''))
count = 0
while(True):
    _,image = cam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.84, 4)
    if len(faces)==0:
        pass
    else:
        for x,y,w,h in faces:
            image = cv2.rectangle(image, (x,y), (x+w,y+h),(0,255,0),3)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
        cv2.imshow("window",image)
        cv2.waitKey(1)
        resized = cv2.resize(roi_gray,(250,250))
        cv2.imwrite(''+user_name+''+str(count)+'.jpg',resized)
        count+=1
    if count==200:
        break
cam.release()
cv2.destroyAllWindows()
    
