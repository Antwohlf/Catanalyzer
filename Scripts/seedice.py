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

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

def gamma_correct(img, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)

def bbox(points, padding=25):
    """
    [xmin xmax]
    [ymin ymax]
    """
    a = np.zeros((2,2))
    a[:,0] = np.min(points, axis=0)
    a[:,1] = np.max(points, axis=0)
    a[0][0] = a[0][0] - padding
    a[0][1] = a[0][1] + padding
    a[1][0] = a[1][0] - padding
    a[1][1] = a[1][1] + padding
    return a

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
        min_threshold = 150 #50                     
        max_threshold = 255 #200                     
        min_area = 60 #100                          
        max_area = 150 #250
        min_circularity = .1
        min_inertia_ratio = .1

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
        im_with_keypoints = cv2.drawKeypoints(gray_image, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        thresh = 50
        X = np.array([list(i.pt) for i in keypoints])
        box = []
        # using hierarchical clustering to cluster the pips so that the pips belonging to
        # different groups could be grouped separately
        num_dict = {}
        if len(X) > 0 and X.shape != (1,2):
            box = bbox(X)
            clusters = hcluster.fclusterdata(X, thresh, criterion="distance")
            num_dict = {np.where(clusters == i)[0][0]:np.sum(clusters==i) for i in np.unique(clusters)}
        return captured_frames[0], num_dict, box

def detect_pips_and_locations_closer(captured_frames):

    """ function to detect the pips on the top face
    and location of each die 
    
    input: list of frames
    output: plot of each frame with detected pips and number of pips
    """
    
    for f in captured_frames:
        gray_image = f

        # setting the parameters for the blob_detection function of OpenCV
        min_threshold = 0 #50                     
        max_threshold = 255 #200                     
        min_area = 100 #400                          
        max_area = 300 #700
        min_circularity = 0.6
        min_inertia_ratio = 0.6

        params = cv2.SimpleBlobDetector_Params()  
        params.filterByArea = True
        params.filterByCircularity = True
        params.filterByInertia = True
        params.filterByColor = True
        params.minThreshold = min_threshold
        params.maxThreshold = max_threshold
        params.minArea = min_area
        params.maxArea = max_area
        params.minCircularity = min_circularity
        params.minInertiaRatio = min_inertia_ratio
        params.minDistBetweenBlobs = 2

        detector = cv2.SimpleBlobDetector_create(params) # create a blob detector object.
        keypoints = detector.detect(gray_image) # keypoints is a list containing the detected blobs.
        # inv_image = cv2.bitwise_not(gray_image)
        # keypoints2 = detector.detect(inv_image)
        im_with_keypoints = cv2.drawKeypoints(gray_image, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        thresh = 70
        X = np.array([list(i.pt) for i in keypoints])
        # using hierarchical clustering to cluster the pips so that the pips belonging to
        # different groups could be grouped separately
        num_dict = {}
        if len(X) > 0 and X.shape != (1,2):
            clusters = hcluster.fclusterdata(X, thresh, criterion="distance")
            num_dict = {np.where(clusters == i)[0][0]:np.sum(clusters==i) for i in np.unique(clusters)}
        return im_with_keypoints, num_dict

def run_stats_dice(dice_states, var_thresh = 300):
    sums = []
    for state in dice_states:
        sums.append(state[0] + state[1])
    var = np.var(sums)
    mode = stats.mode(sums)[0][0]
    max = np.max(sums)
    if var < var_thresh:
        if max == -24:
            return -1, mode, var
        return max, mode, var
    else:
        return -2, mode, var

def get_dice_state(dice_states, frame, kept_states=15, var_thresh=300, penalization=-12, gamma=0.9, show_feed=False):
    # if len(dice_states) == 0:
        # dice_states = [(penalization, penalization)]

    grayscale = gamma_correct(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), gamma);
    frameBuffer = [grayscale]
    output, numDict, box = detect_pips_and_locations(frameBuffer)
    cropped = output

    output2 = frame
    show = False
        
    if len(box) == 2:
        left = int(box[0][0])
        bottom = int(box[1][0])
        top = int(box[1][1])
        right = int(box[0][1])
        cropped = cropped[bottom:top, left:right]
        scale_percent = 200 # percent of original size
        width = int(cropped.shape[1] * scale_percent / 100)
        height = int(cropped.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        cropped = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)
        ret, cropped = cv2.threshold(cropped,90,255,cv2.THRESH_TOZERO)
        cv2.rectangle(frame, (left, bottom), (top, right), (255, 0, 0), 1)
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.dilate(cropped,kernel,iterations = 1)
        show = True
        output2, numDict = detect_pips_and_locations_closer([erosion])

    if(len(numDict) >= 2):
        keys = list(numDict.keys())
        dice_states.append((numDict[keys[0]], numDict[keys[1]]))
    # elif(len(numDict) == 1):
        # keys = list(numDict.keys())
        # dice_states.append((numDict[keys[0]], 0))
    elif(len(numDict) == 0):
        dice_states.append( (penalization, penalization) ) #idfk 

    # Trimming dice states
    if(len(dice_states) > kept_states):
        dice_states.pop(0)

    roll, mode, var = run_stats_dice(dice_states, var_thresh=var_thresh)
    return roll, mode, output2, frame, show, var

def get_dice_state2(dice_states, frame, kept_states=15, var_thresh=300, penalization=-12, gamma=0.9, show_feed=False):
    # if len(dice_states) == 0:
        # dice_states = [(penalization, penalization)]

    grayscale = gamma_correct(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), gamma);
    frameBuffer = [grayscale]
    output, numDict, box = detect_pips_and_locations(frameBuffer)
    cropped = output

    output2 = frame
    if(len(numDict) >= 2):
        keys = list(numDict.keys())
        dice_states.append((numDict[keys[0]], numDict[keys[1]]))
    # elif(len(numDict) == 1):
        # keys = list(numDict.keys())
        # dice_states.append((numDict[keys[0]], 0))
    elif(len(numDict) == 0):
        dice_states.append( (penalization, penalization) ) #idfk 

    # Trimming dice states
    if(len(dice_states) > kept_states):
        dice_states.pop(0)

    roll, mode, var = run_stats_dice(dice_states, var_thresh=var_thresh)
    return roll

# def animate_graph(i):
    # ax1.clear()
    # ax1.plot(variances)

# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)
variances = [0]

def main():
# Setup for streaming
    dice_states = [(-12, -12)]
    dice_states2 = [(-12, -12)]
    windowName = "Live video feed"
    cv2.namedWindow(windowName)
    cap = cv2.VideoCapture(0)

    # Make sure we are capturing

    # ani = animation.FuncAnimation(fig, animate_graph, interval=1000)
    # plt.show(block=False)
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
    
    while ret:
        ret, frame = cap.read()
        # Frame is our image. All processing happens here

        roll, mode, output, frame, show, var = get_dice_state(dice_states, frame)
        roll = get_dice_state2(dice_states2, frame)
        print(dice_states2)
        variances.append(var)
        if roll == -1:
            roll_text = "No roll"
        else:
            roll_text = str(roll)

        #Draw dice text
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,40)
        fontScale              = 0.9
        fontColor              = (255,0,0)
        lineType               = 2
        
        cv2.putText(output, roll_text, 
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