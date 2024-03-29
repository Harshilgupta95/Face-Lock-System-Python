import cv2
import numpy as np
from os import listdir
from os.path import isfile,join
from playsound import playsound

datapath='Dataset/'
onlyfiles=[f for f in listdir(datapath) if isfile(join(datapath,f))]
# print(onlyfiles)
Training_data,Labels = [],[]

for i,files in enumerate(onlyfiles):
    image_path=datapath+onlyfiles[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_data.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)

Labels=np.asarray(Labels, dtype=np.int32)
model=cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_data),np.asarray(Labels))
print('Training Completed')

