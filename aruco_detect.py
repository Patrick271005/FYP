import time
import cv2
import numpy as np
from picamera2 import Picamera2
#Detects an Aruco marker with the Pi Camera and estimates its distance 
MARKER_SIZE_CM = 16 #width of marker
CALIBRATION_DISTANCE_CM = 30 #Distance from camera to marker
FOCAL_PX = 946 # set after calibration step None
# Setting up the ArUco detection
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())

def marker_pixel_width(corner):
    #returns the markers apparent calculated width in pixels
    pts = corner.reshape(4,2) #four xy corners
    sides =[np.linalg.norm(pts[i] -pts[(i+1)%4]) for i in range(4)]
    #subtracts one corner from the next to give the distance, np.linalg.norm turns that distance to length and averages it
    return float(np.mean(sides)) 

#start the camera
picam2=Picamera2()
config = picam2.create_video_configuration(
    main={"size":(1280,720),"format":"RGB888"}
)
picam2.configure(config)
print(picam2.camera_configuration()["main"])
picam2.start()
time.sleep(1)

saved = False
suggestions =[]
for _ in range(60):
    frame = picam2.capture_array("main")
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)

    if ids is not None:
        height, width = gray.shape
        for corner, marker_id in zip(corners,ids.flatten()):
            px = marker_pixel_width(corner)
            centre_x = corner.reshape(4,2)[:,0].mean()
            offset = centre_x - width /2 

            if FOCAL_PX is None:
                suggested = px * CALIBRATION_DISTANCE_CM / MARKER_SIZE_CM
                suggestions.append(suggested)
                print(f"ID {marker_id}: {px:.1f} px wide -> FOCAL_PX = {suggested:.0f}"
                      f"(if the marker is {CALIBRATION_DISTANCE_CM} cm away)")

            else:
                distance = MARKER_SIZE_CM * FOCAL_PX/ px
                print(f"ID{marker_id}: {distance:.1f} cm away, {offset:+.0f} px from centre")
        if not saved:
            aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.imwrite("aruco_detected.jpg",frame)
            saved=True
    time.sleep(0.2)
if suggestions:
    s = sorted(suggestions)
    print(f"{len(s)} detections: FOCAL_PX min {s[0]:.0f},"
          f"median{s[len(s) //2]:.0f}, max{s[-1]:.0f}")
    
picam2.stop()
picam2.close()
if not saved:
    print("no marker found")

