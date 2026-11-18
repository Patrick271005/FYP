import cv2

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

#Marker ID 0, draws 600 x 600 pixels
marker = aruco.generateImageMarker(dictionary, 1, 600) #changes middle value for marker ids

#add a white border to make marker visible on paper
marker = cv2.copyMakeBorder(marker, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)

cv2.imwrite("marker_id1.png", marker)
print("Saved marker_id1.png")