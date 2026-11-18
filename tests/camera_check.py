from picamera2 import Picamera2
#Check what camera is detected 
print("Cameras found:", Picamera2.global_camera_info())

#Create camera and set it for a full size photo
picam2 = Picamera2()
config = picam2.create_still_configuration()
picam2.configure(config)

#activate camera take a photo and deactivate 
picam2.start()
picam2.capture_file("camera_check.jpg")
picam2.stop()
picam2.close()

print("Saved camera_check.jpg")