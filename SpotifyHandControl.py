import cv2
import sys
import time
from PIL import ImageGrab
from spotifymacro import spotilib

DELAY = 1
threshold = 4
palm = "haar/palm.xml"
fist = "haar/fist.xml"
def pause_resume():
    spotilib.pause()
    time.sleep(DELAY)
def play_next():
    spotilib.next()
    time.sleep(DELAY)
def main():
    palmCascade = cv2.CascadeClassifier(palm)
    fistCascade = cv2.CascadeClassifier(fist)
    # use counter c to prevent false positive to execute the command
    c = 0
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Trained hyperparameters for the classifier
        palms = palmCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=7, 
            minSize=(80, 80),
            maxSize=(150, 150),
        )
        fists = fistCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=13, 
            minSize=(90, 90),
            maxSize=(170, 170),
        )
        if len(palms) > 0:
            c+=1
            if c > threshold:
                pause_resume()
                c=0
        elif len(fists) > 0:
            c-=1
            if c < -threshold:
                play_next()
                c=0     
        else:
            c = int(c/2)
##      ### for debugging purpose  ###
            
##        for (x, y, w, h) in palms:
##            print(str(w) + ' ,' + str(h))
##            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
##        for (x, y, w, h) in fists:
##            print(str(w) + ' ,' + str(h))
##            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
##        cv2.imshow('Video', frame)
##        print(c)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
