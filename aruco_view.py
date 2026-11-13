# aruco_view.py
# Live debugging view: streams what the camera sees, with detected markers
# outlined and their distance written on them, to a web page.
# Open http://pf-fyp-pi.local:8000 in a browser on the laptop. Ctrl+C to stop.

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from picamera2 import Picamera2

# ---- Numbers from the calibration file ---------------------------------
MARKER_SIZE_CM = 16.0
FOCAL_PX = 946
PORT = 8000

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())


def marker_pixel_width(corner):
    pts = corner.reshape(4, 2)
    sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    return float(np.mean(sides))


# ---- Shared between the camera loop and the web server -----------------
latest_jpeg = None
lock = threading.Lock()

PAGE = (b"<html><body style='margin:0;background:#111'>"
        b"<img src='/stream' style='width:100%'></body></html>")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with lock:
                        jpg = latest_jpeg
                    if jpg is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(jpg)).encode()
                                         + b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser tab was closed
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass  # keep the terminal quiet


server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()

# ---- Camera (same settings as the calibration run) ---------------------
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(1)

print(f"Streaming on port {PORT}. Open http://pf-fyp-pi.local:{PORT}. Ctrl+C to stop.")
try:
    while True:
        frame = picam2.capture_array("main")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            for corner, marker_id in zip(corners, ids.flatten()):
                distance = MARKER_SIZE_CM * FOCAL_PX / marker_pixel_width(corner)
                x, y = corner.reshape(4, 2)[0]   # the marker's first corner
                cv2.putText(frame, f"{distance:.1f} cm", (int(x), max(int(y) - 12, 24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        small = cv2.resize(frame, (640, 360))   # smaller picture = less work and lag
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with lock:
                latest_jpeg = jpg.tobytes()
except KeyboardInterrupt:
    print("Stopped")
finally:
    server.shutdown()
    picam2.stop()
    picam2.close()