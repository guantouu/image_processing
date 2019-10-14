import os
import threading
import sqlite3
import numpy as np
from keras.models import load_model
from face_recognition import face_recognition
from flask import Flask, request, make_response, render_template
app = Flask('__name__')
file_path = './static'
SQLITE_DB_PATH = '../database.db'

@app.route('/upload', methods=['POST'])
def upload():
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    my_file = request.files['upload']
    my_file.save(os.path.join(file_path, my_file.filename))
    t = threading.Thread(target=face_recognition,  
          args=(my_file.filename, )
        )
    t.start()
    return ''

@app.route('/index')
def index():
    db = sqlite3.connect(SQLITE_DB_PATH)
    with db:
        results = db.execute("SELECT * FROM detection").fetchall()
    db.close()
    return render_template(
        'index.html', results=results
    )

        

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)