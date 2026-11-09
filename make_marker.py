import cv2

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

#Marker ID 0, draws 600 x 600 pixels
marker = aruco.generateImageMarker(dictionary, 0, 600)

#add a white border to make marker visible on paper
marker = cv2.copyMakeBorder(marker, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)

cv2.imwrite("marker_id0.png", marker)
print("Saved marker_id0.png")