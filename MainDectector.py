
import cv2
import traceback
import time
import cv2, queue, threading, time
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import numpy as np
import numpy
resnet = InceptionResnetV1(pretrained='vggface2').eval()
import torchvision.transforms as transforms
from sklearn import svm
import pickle
from joblib import Parallel, delayed
import joblib
from skimage.transform import resize
model = joblib.load(r'C:\\Users\Vriska S\Desktop\MMS project\dumpnewardu.pkl')
mtcnn = MTCNN(    image_size=160, margin=0, min_face_size=40,
    thresholds=[0.6, 0.6, 0.7], factor=0.709, post_process=True,keep_all=True)
print(model)

NameDic ={ 0: "Gagan", 1: "Akhilesh", 2: "Pranav", 3: "Anirudh"}


# bufferless VideoCapture
class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


o = []
for q in range(0, 1):
    #frame = cam.read()
    frame = cv2.imread(r"C:\Users\Vriska S\Desktop\out1.jpg", cv2.IMREAD_COLOR)
    alpha = 1  # Contrast control (1.0-3.0)
    beta = 50  # Brightness control (0-100)
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    boxes = mtcnn(frame)
    for box in boxes:
        frame = box.unsqueeze(0)
        try:

            img_embedding = resnet(frame)
            new = img_embedding.cpu().detach().numpy()
            numberID= model.predict(new)
            print(numberID)
            o.append(numberID[0])
        except:
            traceback.print_exc()
            print(len(frame))
        print('discarded')
    # cv2.imshow('video', frame)
name = []
print(o)
for num in o :
    name.append(NameDic[num])

print(name)

cv2.destroyAllWindows()
