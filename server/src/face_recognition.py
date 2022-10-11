import sys, os, glob, time
import cv2
import sqlite3
import numpy as np
from skimage.transform import resize
from keras.models import load_model
from keras import backend as K 

faces_folder_path = "../rec"
file_path = './static'
SQLITE_DB_PATH = '../database.db'
cascade_path = "../data/haarcascade_frontalface_default.xml"
model_path = '../data/facenet_keras.h5'
model = load_model(model_path)
model.predict(np.zeros((1, 160, 160, 3)))
image_size = 160
descriptors = []
candidate = []

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size 
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def align_image(img, margin):
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    if(len(faces)>0):
        (x, y, w, h) = faces[0]
        face = img[y:y+h, x:x+w]
        faceMargin = np.zeros((h+margin*2, w+margin*2, 3), dtype = "uint8")
        faceMargin[margin:margin+h, margin:margin+w] = face
        aligned = resize(faceMargin, (image_size, image_size), mode='reflect')
        return aligned
    else:
        return None

def preProcess(img):
    whitenImg = prewhiten(img)
    whitenImg = whitenImg[np.newaxis, :]
    return whitenImg

def face_recognition(filenames):
    who = ''
    for member in glob.glob(os.path.join(faces_folder_path, "*.jpeg")):
        base = os.path.basename(member)
        candidate.append(os.path.splitext(base)[0])
        aligned = align_image(cv2.imread(member), 6)
        if(aligned is not None):
            faceImg = preProcess(aligned)
            embs = l2_normalize(np.concatenate(model.predict(faceImg)))
            descriptors.append(embs)
    aligned = align_image(cv2.imread(os.path.join(file_path, filenames)), 6)
    dist = []
    if(aligned is not None):
        faceImg = preProcess(aligned)
        embs_valid = l2_normalize(np.concatenate(model.predict(faceImg)))
        for x in descriptors:
            dist_ = np.linalg.norm(x - embs_valid)
            dist.append(dist_)
    c_d = dict(zip(candidate,dist))
    print(c_d)
    for key, value in sorted(c_d.items(), key=lambda d:d[1]):
        if value <= 0.6:
            who = key
            break
        else:
            who ='guest_' + filenames.split('.', 1)[0]

    db =sqlite3.connect(SQLITE_DB_PATH)
    with db:
        db.execute('INSERT INTO detection (file, name)'  \
                  +'VALUES (?, ?)', (filenames, who))
    db.close()