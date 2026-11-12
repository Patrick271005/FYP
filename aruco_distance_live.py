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
picam2.start()
time.sleep(1)

#takes last 5 distances
recent = deque(maxlen=5)
print("Running distance CTRL C to stop")
try:
    while True:
        frame = picam2.capture_array("main")
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = detector.detectMarkers(gray)

        if ids is not None:
            height, width = gray.shape
            for corner, marker_id in zip(corners, ids.flatten()):
                px = marker_pixel_width(corner)
                centre_x = corner.reshape(4,2)[:,0].mean()
                offset = centre_x - width / 2

                distance = MARKER_SIZE_CM * FOCAL_PX / px
                recent.append(distance)
                average = sum(recent) / len(recent)
                print(f"ID {marker_id}: {distance:.1f} cm  (avg {average:.1f}), " 
                      f"{offset:+.0f} px from centre")
        else:
            recent.clear()
            print("No marker in view")
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Stopped")
finally:
    picam2.stop()
    picam2.close()


