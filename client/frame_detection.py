import cv2
import dlib
import time
import requests 
import os
from datetime import datetime
import numpy as np
import multiprocessing as mp
URL = 'http://flask.guantouu.pw/upload'
file_path = './picture'
class FrameDetection():
    def initialize(self, frame01):
        blur = cv2.GaussianBlur(frame01, (3, 3), 1.5)
        grayblur = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
        ret, threshold1 = cv2.threshold(grayblur, 40, 255, cv2.THRESH_BINARY_INV)
        canny= cv2.Canny(threshold1, 50, 150, apertureSize=3)
        while True:
            cv2.imshow('display', canny)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                break
        try:
            lines = cv2.HoughLines(canny, 1, (np.pi / 180), 93)
            border_right, border_Left, border_top = 0, canny.shape[1], canny.shape[0]
            horizon_rho, horizon_theta, right_rho, right_theta, left_rho, left_theta = 0, 0, 0, 0, 0, 0
            lineX = lines[:,0,:]  # test don't change var
            for rho, theta in lineX: 
                if (theta < (np.pi/4.0)) or (theta > (3*np.pi/4.0)):
                    X = rho/np.cos(theta)
                    if X > border_right:
                        border_right = X
                        right_rho = rho
                        right_theta = theta
                    elif X < border_Left:
                        border_Left = X
                        left_rho = rho
                        left_theta =theta
                elif (theta > (np.pi/6.0)) or (theta < (4*np.pi/6.0)):
                    Y = rho/np.sin(theta)
                    if Y < border_top :
                        border_top = Y
                        horizon_rho = rho
                        horizon_theta = theta
            door_point = {
                'right_point1' : (int(right_rho / np.cos(right_theta)), 0),
                'right_point2' : (int((right_rho - canny.shape[0] * np.sin(right_theta))/np.cos(right_theta)), canny.shape[0]),
                'left_point1' : (int(left_rho / np.cos(left_theta)), 0),
                'left_point2' : (int((left_rho - canny.shape[0] * np.sin(left_theta))/np.cos(left_theta)), canny.shape[0]),
                'top_point1' : (0, int(horizon_rho/np.sin(horizon_theta))),
                'top_point2' : (canny.shape[1], int((horizon_rho - canny.shape[1] * np.cos(horizon_theta))/np.sin(horizon_theta)))
            }
            return door_point
        except:  #record which except will happen
            return False
    
    def door_detection(self, frame01, door_queues, door_point, door_status):
        while True :
            count1, count2 = 0, 0
            frame = door_queues.get()
            diff = cv2.absdiff(frame01, frame)
            for  R, G, B  in diff[(door_point['top_point1'][1] + 50),  \
                             (door_point['left_point1'][0] - 10):(door_point['right_point2'][0] - 10)]:
                if door_status.value == 0:
                    if G > 40:
                        count1 += 1
                        if count1 > 80:
                            door_status.value = 1
                else:
                    if G <= 20:
                        count2 += 1
                        if count2 > 140:
                            door_status.value = 0

    def face_detection(self, face_queues, door_status, location_queues):
        detector = dlib.get_frontal_face_detector() #understand about detector(dlib)
        while True :
            if door_status.value == 0 or face_queues.empty() == True:
                time.sleep(0.01)
                continue
            frame = face_queues.get()
            face_rects, scores, idx = detector.run(frame, 0)
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                if scores[i] >= 0.60:
                    square_point = [x1 , y1 , x2, y2]
                    location_queues.put(square_point)
                    location_queues.get() if location_queues.qsize() > 1 else time.sleep(0.01)
                    self.upload_detection(frame)

    def upload_detection(self, frame):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name =file_path + '/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.jpeg'
        cv2.imwrite(file_name, frame)
        picture = {'upload':open(file_name, 'rb')}
        r = requests.post(URL, files=picture)


