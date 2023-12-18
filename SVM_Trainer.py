
import os

import numpy
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import time
import torch
import numpy as np
from PIL import Image,ImageDraw
from sklearn.svm import SVC
import pickle
from joblib import Parallel, delayed
import joblib
from sklearn.gaussian_process.kernels import RBF
from retinaface import RetinaFace
import cv2, queue, threading, time
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from deepface import DeepFace
models = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace",
  "DeepID",
  "ArcFace",
  "Dlib",
  "SFace",
]

backends = [
  'opencv',
  'ssd',
  'dlib',
  'mtcnn',
  'retinaface',
  'mediapipe'
]


clf = xgb.XGBClassifier()
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
# folder path

mtcnn = MTCNN(    image_size=160, margin=0, min_face_size=40,
    thresholds=[0.3, 0.5, 0.5], factor=0.709, post_process=True,) #used because better performance
resnet = InceptionResnetV1(pretrained='vggface2').eval()

#list to store files
facelist=[]
target =[]
for n in range (0,8):
    dir_path = r"C:\Users\Vriska S\Desktop\train\\" + str(n)
    res = []
# Iterate directory
    for path in os.listdir (dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            res.append(path)
    print(res)
    for item in res:
        print(item)
        frame = cv2.imread(str(r'C:\Users\Vriska S\Desktop\train' + str('\\') + str(n) +str('\\')+str(item)), cv2.IMREAD_COLOR)
        alpha = 1  # Contrast control (1.0-3.0)
        beta = 70  # Brightness control (0-100)
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        box = mtcnn(frame)
        frame = box.unsqueeze(0)
        img_embedding = resnet(frame)
        #dfs = DeepFace.represent(frame, model_name=models[6],
        #                         detector_backend=backends[4])
        #img_embedding=dfs[0]['embedding']
        facelist.append(img_embedding)
        target.append(n)


new= []
for a in facelist:
   new.append(a.detach().cpu().numpy()) #UNCOMMENT FOR FACENET
new = [np.concatenate(i) for i in new] #UNCOMMENT FOR FACENET
clf = SVC(probability=True, kernel='rbf', gamma= 1,C=2)
resultz = [new,target]
joblib.dump(resultz,"C:\\Users\Vriska S\Desktop\MMS Project\data.pkl")
clf.fit(new,target) #TURN (new,target) for FACENET, (facelist,target) for arcnet
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
grid.fit(new, target)
print(grid.best_params_)
joblib.dump(clf,"C:\\Users\Vriska S\Desktop\MMS Project\dumpnewardu.pkl")
print("done")
