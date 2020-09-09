import numpy as np
import cv2
import os


recognizer = cv2.face.LBPHFaceRecognizer_create()

path = os.getcwd()

data_path = os.path.join(path,'images')

IDS = []
Faces =[]
ddd = {}

count=1

for files,roots,dirs in os.walk(data_path):#file will have path first time path till data_path and from next time path till first folder and so on
                                           #roots will have all folders in tha path (files)
                                           #dirs will have all the data(img) in path(files)
    #print(files)
    
    for root in roots:
        ddd[root]=count
        count+=1
    
    for di in dirs:
        imagep = os.path.join(files,di)
        #print(imagep)
        #print(files,di)
        
        if imagep.endswith('.jpg'):
            #print(files)
            ids = files.split("\\")[2]
            IDS.append(ddd[ids])
            #print(IDS)
            image = cv2.imread(imagep)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(image,(250,250))
            Npface = np.array(resized,'uint8')
            Faces.append(Npface)
#print(ddd)
IDS = np.array(IDS)
print(Faces)
recognizer.train(Faces,IDS)
os.chdir(path)
recognizer.save('trained_data.yml')
