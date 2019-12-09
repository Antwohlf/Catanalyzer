import json
import numpy as np
import os
import cv2
from matplotlib.image import imsave
import matplotlib.pyplot as plt

# A, B
blue_avg = (130, 95)
red_avg = (189, 160)
white_avg = (130, 130)
orange_avg = (173, 180)

color_classes = [blue_avg, red_avg, white_avg, orange_avg]

hsv_classes = [106, 178, 18, 7]

names = ["blue", "red", "white", "orange", "unknown"]

def detect_color_lab(img, cropStart=4, toleranceStart=13):
    cx = img.shape[0] // 2
    cy = img.shape[1] // 2
    crop = cropStart
    tolerance = toleranceStart
    class_index = 4

    while(class_index == 4 and crop < 20):
        crop = crop + 1
        left = cx - crop
        right = cx + crop
        top = cy - crop
        bottom = cy + crop
        cropped = img[left:right, top:bottom]
        lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        a_mean = np.mean(a)
        b_mean = np.mean(b)

        for i, color in enumerate(color_classes):
            a_diff = abs(color[0] - a_mean)
            b_diff = abs(color[1] - b_mean)
            if a_diff < tolerance and b_diff < tolerance:
                class_index = i

    if class_index == 4:           
        #if we still haven't found it, increase tolerance
        while(class_index == 4 and tolerance < 50):
            tolerance = tolerance + 1
            left = cx - crop
            right = cx + crop
            top = cy - crop
            bottom = cy + crop
            cropped = img[left:right, top:bottom]
            lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            a_mean = np.mean(a)
            b_mean = np.mean(b)

            for i, color in enumerate(color_classes):
                a_diff = abs(color[0] - a_mean)
                b_diff = abs(color[1] - b_mean)
                if a_diff < tolerance and b_diff < tolerance:
                    class_index = i
    return class_index

def detect_color_hsv(img, cropStart=4, toleranceStart=2):
    cx = img.shape[0] // 2
    cy = img.shape[1] // 2
    crop = cropStart
    tolerance = toleranceStart
    class_index = 4

    
    while(class_index == 4 and crop < 20):
        crop = crop + 1
        left = cx - crop
        right = cx + crop
        top = cy - crop
        bottom = cy + crop
        cropped = img[left:right, top:bottom]
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h_mean = np.mean(h)

        for i, color in enumerate(hsv_classes):
            h_diff = abs(color - h_mean)
            if h_diff < tolerance:
                class_index = i

    if class_index == 4:           
        #if we still haven't found it, increase tolerance
        while(class_index == 4 and tolerance < 10):
            tolerance = tolerance + 1
            left = cx - crop
            right = cx + crop
            top = cy - crop
            bottom = cy + crop
            cropped = img[left:right, top:bottom]
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            h_mean = np.mean(h)

            for i, color in enumerate(hsv_classes):
                h_diff = abs(color - h_mean)
                if h_diff < tolerance:
                    class_index = i
    return names[class_index]

def main():
    for i in range(1, 65):
        img = cv2.imread("cropped/catan_28_" + str(i) + ".jpg")
        color = detect_color_lab(img)
        print(str(i) + ": " + color)

if __name__ == "__main__":
    main()