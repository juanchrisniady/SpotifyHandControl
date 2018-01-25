import cv2
import sys
import time
from PIL import ImageGrab
from spotifymacro import spotilib

DELAY = 1
threshold = 4
palm = "haar/palm.xml"
fist = "haar/fist.xml"
palmCascade = cv2.CascadeClassifier(palm)
fistCascade = cv2.CascadeClassifier(fist)
# use counter c to prevent false positive to execute the command
c = 0
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
## stop    
    palms = palmCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=3, #4
        minSize=(80, 80),
    )
## prev
    fists = fistCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=3, #10
        minSize=(80, 80),
    )
    if len(fists) > 0:
        c-=1
        if c < -threshold:
            spotilib.next()
            time.sleep(DELAY)
            c=0
    elif len(palms) > 0:
        c+=1
        if c > threshold:
            spotilib.pause()
            time.sleep(DELAY)
            c=0
    
    else:
        c = int(c/2)
    print(c)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
