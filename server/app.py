import os
import dlib
import threading
from face_recognition import face_recognition
from flask import Flask, request, make_response
app = Flask('__name__')
file_path = './picture'

detector = dlib.get_frontal_face_detector()
face_feature = dlib.shape_predictor(
    './data/shape_predictor_68_face_landmarks.dat'
)
face_rec = dlib.face_recognition_model_v1(
    './data/dlib_face_recognition_resnet_model_v1.dat'
)

@app.route('/upload', methods=['POST'])
def upload():
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    my_file = request.files['upload']
    my_file.save(os.path.join(file_path, my_file.filename))
    t = threading.Thread(target=face_recognition,  \
          args=(my_file.filename, detector, face_feature, face_rec)
        )
    t.start()
    return ''

@app.route('/show/<string:filename>', methods=['GET'])
def show(filename):
    image = open(os.path.join(file_path, filename), 'rb').read()
    response = make_response(image)
    response.headers['content-Type'] = 'image/jpg'
    return response

        

if __name__ == "__main__":
    app.run()