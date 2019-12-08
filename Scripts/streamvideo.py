import cv2
import numpy as np

def startStream(name="Live Video Feed", cam=0):
    windowName = name
    cv2.namedWindow(windowName)
    cap = cv2.VideoCapture(cam)

    # Make sure we are capturing
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    return ret, cap

def displayStream(img, name="Live Video Feed"):
    cv2.imshow(name, img)
    if cv2.waitKey(1) == 27:
        return True
    return False

def endStream(cap, name="Live Video Feed"):
    cv2.destroyWindow(name)
    cap.release()


def main():
    ret, cap = startStream()
    stopFeed = False
    while ret and not stopFeed:
        ret, frame = cap.read()
        # Frame is our image. All processing happens here
        output = cv2.Canny(frame, 100, 200)
        stopFeed = displayStream(output)
    # Teardown 
    endStream()

if __name__ == "__main__":
    main()