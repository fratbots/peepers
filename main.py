import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

propFps = cap.get(cv2.CAP_PROP_FPS)
print("CAP_PROP_FPS: {0}".format(propFps))

realFps = 0
frameNum = 0
timeStart = time.time()

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = ret

    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frameNum = frameNum + 1
    timeNow = time.time()
    realFps = frameNum / (timeNow - timeStart)

    if frameNum % 30 == 0:
        print("FPS: {0}".format(realFps))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
