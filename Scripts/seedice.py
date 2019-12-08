import os
import cv2
import numpy as np
from os import listdir

import time
import math
import matplotlib
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
import scipy.stats as stats

def gamma_correct(img, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)

def detect_pips_and_locations(captured_frames):
    """ function to detect the pips on the top face
    and location of each die 
    
    input: list of frames
    output: plot of each frame with detected pips and number of pips
    """
    
    for f in captured_frames:
        gray_image = f
        x_range1 = int(gray_image.shape[0]*0.06)
        x_range2 = int(gray_image.shape[0]*0.91)
        y_range1 = int(gray_image.shape[1]*0.05)
        y_range2 = int(gray_image.shape[1]*0.95)

        # cropping out the outer border
        gray_image[:,0:y_range1] = 0.0
        gray_image[:,y_range2:] = 0.0
        gray_image[:x_range1,:] = 0.0
        gray_image[x_range2:,:] = 0.0

        # setting the parameters for the blob_detection function of OpenCV
        min_threshold = 170 #50                     
        max_threshold = 250 #200                     
        min_area = 60 #100                          
        max_area = 250 #250
        min_circularity = .7
        min_inertia_ratio = 0.0

        params = cv2.SimpleBlobDetector_Params()  
        params.filterByArea = True
        params.filterByCircularity = True
        params.filterByInertia = True
        params.minThreshold = min_threshold
        params.maxThreshold = max_threshold
        params.minArea = min_area
        params.maxArea = max_area
        params.minCircularity = min_circularity
        params.minInertiaRatio = min_inertia_ratio

        detector = cv2.SimpleBlobDetector_create(params) # create a blob detector object.
        keypoints = detector.detect(gray_image) # keypoints is a list containing the detected blobs.
        # inv_image = cv2.bitwise_not(gray_image)
        # keypoints2 = detector.detect(inv_image)
        im_with_keypoints = cv2.drawKeypoints(gray_image, keypoints, np.array([]), (255, 0, 0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        thresh = 50
        X = np.array([list(i.pt) for i in keypoints])
        # using hierarchical clustering to cluster the pips so that the pips belonging to
        # different groups could be grouped separately
        num_dict = {}
        if len(X) > 0 and X.shape != (1,2):
            clusters = hcluster.fclusterdata(X, thresh, criterion="distance")
            cluster_no = [np.sum(clusters==i) for i in clusters]
            num_dict = {np.where(clusters == i)[0][0]:np.sum(clusters==i) for i in np.unique(clusters)}
        return im_with_keypoints, num_dict

def run_stats_dice(dice_states, var_thresh = 170):
    sums = []
    for state in dice_states:
        sums.append(state[0] + state[1])
    var = np.var(sums)
    mode = stats.mode(sums)
    if var < var_thresh:
        if(mode[0] == -12):
            return -1
        return mode[0]
    else:
        return -1


def main():
# Setup for streaming

    dice_states = []
    kept_states = 20

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
        grayscale = gamma_correct(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0.15);
        frameBuffer = [grayscale]
        frameBuffer.append(grayscale)
        output, numDict = detect_pips_and_locations(frameBuffer)

        if(len(numDict) == 2):
            keys = list(numDict.keys())
            dice_states.append((numDict[keys[0]], numDict[keys[1]]))
        elif(len(numDict) == 0):
            dice_states.append( (-12, -12) ) #idfk 

        # Trimming dice states
        if(len(dice_states) > kept_states):
                dice_states.pop(0)

        mode = run_stats_dice(dice_states)
        mode_text = str(mode)
        if mode_text == "[-24]":
            mode_text = "No roll"

        #Draw dice text
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale              = 0.8
        fontColor              = (255,255,255)
        lineType               = 2
        
        cv2.putText(output, str(mode_text), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        cv2.imshow(windowName, cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) == 27:
            break
        
    # Teardown 
    cv2.destroyWindow(windowName)
    cap.release()

if __name__ == "__main__":
    main()