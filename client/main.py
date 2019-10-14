import cv2
import dlib
import multiprocessing as mp
from frame_interface import FrameInterface
from frame_detection import FrameDetection

class MainProcess():
    def __init__(self):
        self.frame_detection = FrameDetection()
        self.frame_interface = FrameInterface()
    
    def process_controller(self):
        cap = cv2.VideoCapture(0) #require raspberry pi camera port serial
        cap.set(3, 640)
        cap.set(4, 480)
        ret, frame01 = cap.read()
        door_point = self.frame_detection.initialize(frame01)
        if door_point is not False: 
            door_status = mp.Value('i', 0)
            door_queues = mp.Queue(maxsize=4)
            face_queues = mp.Queue(maxsize=4)
            location_queues = mp.Queue(maxsize=4)
            output_queues = mp.Queue(maxsize=4)
            entrance = mp.Process(
                target=self.frame_interface.entrance, \
                args=(cap, door_queues, face_queues, output_queues)
                )
            output = mp.Process(
                target=self.frame_interface.output,  \
                args=(output_queues, location_queues, door_point)
                )
            door_detection = mp.Process(
                target=self.frame_detection.door_detection,  \
                args=(frame01, door_queues, door_point, door_status)
                )
            face_detection = mp.Process(
                target=self.frame_detection.face_detection,  \
                args=(face_queues, door_status, location_queues)
                )
            entrance.start()
            output.start()
            door_detection.start()
            face_detection.start()
            while True :
                if output.is_alive() is False:
                    entrance.terminate()
                    door_detection.terminate()
                    face_detection.terminate()
                    break
        else :
            pass
#           while True:
#               cv2.imshow('display' , frame01)
#               if  cv2.waitKey(1) == 27 :
#                   cv2.destroyAllWindows()
#                   break
        cap.release()
        
if __name__ == '__main__':
    main = MainProcess()
    main.process_controller()
    
    
