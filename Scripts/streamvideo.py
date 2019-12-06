import cv2
import numpy as np

def main():
    # Setup for streaming
    windowName = "Live video feed"
    cv2.namedWindow(windowName)
    cap = cv2.VideoCapture(1)

    # Make sure we are capturing
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
    
    while ret:
        ret, frame = cap.read()
        # Frame is our image. All processing happens here
        output = cv2.Canny(frame, 100, 200)
        zeros = np.zeros_like(output)
        coolput = np.dstack( (zeros, output, zeros))
        newout = cv2.addWeighted(coolput, 1, frame, 1, 0.0)
        cv2.imshow(windowName, newout)
        if cv2.waitKey(1) == 27:
            break
        
    # Teardown 
    cv2.destroyWindow(windowName)
    cap.release()

if __name__ == "__main__":
    main()