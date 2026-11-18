import time
from collections import deque
import cv2
import numpy as np
from picamera2 import Picamera2

#values from calibration file
MARKER_SIZE_CM={
    0: 16.0, #big marker
    1: 3.75, #small marker
} 
FOCAL_PX=946

aruco=cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())

def marker_pixel_width(corner):
    #returns the markers apparent width in pixels by averaging its 4 sides
    pts = corner.reshape(4,2)
    sides = [np.linalg.norm(pts[i] -pts[(i+1)%4]) for i in range(4)]
    return sum(sides)/4

picam2 = Picamera2()
config = picam2.create_video_configuration( main ={"size":(1280,720), "format":"RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(1)

#Average for each marker
recent = {}
print("Running distance CTRL C to stop")
try:
    while True:
        frame = picam2.capture_array("main")
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = detector.detectMarkers(gray)

        seen = set()

        if ids is not None:
            for corner, marker_id in zip(corners, ids.flatten()):
                marker_id = int(marker_id)
                size_cm = MARKER_SIZE_CM.get(marker_id)
                if size_cm is None:
                    continue #If Id is not in the table for detection ignore it


                px = marker_pixel_width(corner)
                centre_x = corner.reshape(4,2)[:,0].mean()
                offset = centre_x - frame.shape[1] / 2
                distance = size_cm * FOCAL_PX / px

                history = recent.setdefault(marker_id, deque(maxlen=5))
                history.append(distance)
                average = sum(history)/ len(history)

                seen.add(marker_id)
                print(f"ID {marker_id}: {distance:.1f} cm  (avg {average:.1f}), " 
                      f"{offset:+.0f} px from centre")
        for marker_id in list(recent):
            if marker_id not in seen:
                del recent[marker_id]
        if not seen:
            print("no marker in view")
            
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Stopped")
finally:
    picam2.stop()
    picam2.close()


