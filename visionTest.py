#Based off of code from: https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
import argparse
import imutils
import time
import cv2
from imutils.video import VideoStream

vs = VideoStream(src=0).start()
time.sleep(2.0)

firstFrame = None

while True:
    frame = vs.read()

    if frame is None:
        break

    frame = imutils.resize(frame, width = 500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gray
        continue

    deltaFrame = cv2.absdiff(firstFrame, gray)
    
    thresh = cv2.threshold(deltaFrame, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations = 2)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for c in contours:
        if cv2.contourArea(c) > 5000 or cv2.contourArea(c) < 1000:
            continue
        (x, y, z, t) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + z, y + t), (0, 0, 255), 2)

    cv2.imshow("Vision Tracking Test", frame)
    cv2.imshow("Frame Delta", deltaFrame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
