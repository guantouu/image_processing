import cv2
import time
import multiprocessing as mp

class FrameInterface():
    def entrance(self, cap, door_queues, face_queues, output_queues):
        while True :
            ret, frame = cap.read()
            door_queues.put(frame)
            face_queues.put(frame)
            output_queues.put(frame)
            door_queues.get() if door_queues.qsize() > 1 else time.sleep(0.01)
            face_queues.get() if face_queues.qsize() > 1 else time.sleep(0.01)
            output_queues.get() if output_queues.qsize() > 1 else time.sleep(0.01)

    def output(self, output_queues, location_queues, door_point):
        while True :
            frame = output_queues.get()
            if location_queues.empty() == False:
                square_point = location_queues.get()
                cv2.rectangle(frame, (square_point[0], square_point[1]),  \
                                     (square_point[2], square_point[3]),  \
                                     (0,255,0), 4, cv2.LINE_AA)
            cv2.line (frame, (door_point['left_point2'][0]-10, door_point['top_point1'][1]+50),  \
                             (door_point['right_point1'][0]-10, door_point['top_point1'][1]+50),  \
                             (0,255,55), 2)
            cv2.line(frame, door_point['right_point1'], door_point['right_point2'], (0,255,0), 2)
            cv2.line(frame, door_point['left_point1'], door_point['left_point2'], (0,255,0), 2)
            cv2.line(frame, door_point['top_point1'], door_point['top_point2'], (0,255,0), 2)
            cv2.imshow('display' , frame)
            if  cv2.waitKey(1) == 27 :
                cv2.destroyAllWindows()
                break
