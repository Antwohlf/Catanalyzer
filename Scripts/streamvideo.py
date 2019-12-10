import cv2
import numpy as np
from makeOverlay import makeOverlay
import PIL

def startStream(name="Live Video Feed", cam=1):
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

def pil2cv(img):
    pil_image = img.convert('RGB') 
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    return open_cv_image

def main():
    ret, cap = startStream()
    stopFeed = False
    features = {"dice": -1, "red":{"road":9, "town":3, "city":2}, "white":{"road":5, "town":2, "city":0}, "blue":{"road":12, "town":4, "city":2}, "orange":{"road":3, "town":4, "city":5}}
    while ret and not stopFeed:
        ret, frame = cap.read()
        # Frame is our image. All processing happens here
        output = makeOverlay(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA), features)
        stopFeed = displayStream(pil2cv(output))
    # Teardown 
    endStream()

if __name__ == "__main__":
    main()