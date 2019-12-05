import cv2

def main():
    # Setup for streaming
    windowName = "Live video feed"
    cv2.namedWindow(windowName)
    cap = cv2.VideoCapture(0)

    # Make sure we are capturing
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
    
    while ret:
        ret, frame = cap.read()
        # Frame is our image. All processing happens here
        output = frame#cv2.Canny(frame, 100, 200)
        cv2.imshow(windowName, output)
        if cv2.waitKey(1) == 27:
            break
        
    # Teardown 
    cv2.destroyWindow(windowName)
    cap.release()

if __name__ == "__main__":
    main()