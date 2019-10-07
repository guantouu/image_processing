import sys,os,dlib,glob,numpy
import cv2
import sqlite3
faces_folder_path = "../rec"
file_path = './static'
SQLITE_DB_PATH = '../database.db'
descriptors = []
candidate = []

def face_recognition(filenames, detector, face_feature, face_rec):
    who = ''
    detector = dlib.get_frontal_face_detector()
    picture = cv2.imread(os.path.join(file_path, filenames))
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        base = os.path.basename(f)
        candidate.append(os.path.splitext(base)[0])
        img = cv2.imread(f)
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            shape = face_feature(img, d)
            face_descriptor = face_rec.compute_face_descriptor(img, shape)
            v = numpy.array(face_descriptor)
            descriptors.append(v)
    dets = detector(picture, 1)
    dist = []
    for k, d in enumerate(dets):
        shape = face_feature(picture, d)
        face_descriptor = face_rec.compute_face_descriptor(picture, shape)
        d_test = numpy.array(face_descriptor)
        for i in descriptors:
            dist_ = numpy.linalg.norm(i - d_test)
            dist.append(dist_)
    c_d = dict(zip(candidate,dist))
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
    
