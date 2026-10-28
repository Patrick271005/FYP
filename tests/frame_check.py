import time
from picamera2 import Picamera2

picam2 = Picamera2()

#setting video configuration to size and format OpenCV will use later
conifg = picam2.create_video_configuration(
    main={"size": (640,480), "format": "RGB888"}
)
picam2.configure(conifg)
picam2.start()
time.sleep(1)

frames = 100
start = time.time()
for _ in range(frames):
    frame = picam2.capture_array("main")
elapsed = time.time() - start

print("Frame shape:", frame.shape, "dtype:", frame.dtype)
print(f"{frames} frames in {elapsed:.2f} s = {frames / elapsed:.1f} fps")

picam2.stop()
picam2.close()
