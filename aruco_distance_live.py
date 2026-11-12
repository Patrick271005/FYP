import time
from collections import deque
import cv2
import numpy as np
from picamera2 import Picamera2

#values from calibration file
MARKER_SIZE_CM=16
FOCAL_PX=946

aruco=cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())

def marker_pixel_width(corner):
    #returns the markers apparent width in pixels by averaging its 4 sides
    pts = corner.reshape(4,2)
    sides = [np.linalg.norm(pts[i] -pts[(i+1)%4]) for i in range(4)]
    return float(np.mean(sides))

picam2 = Picamera2()
config = picam2.create_video_configuration( main ={"size":(1280,720), "format":"RGB888"})
picam2.configure(config)
time.sleep(1)