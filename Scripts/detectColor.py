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

    while(class_index == 4 and crop < 10):
        crop = crop + 1
        left = cx - crop
        right = cx + crop
        top = cy - crop
        bottom = cy + crop
        cropped = img[left:right, top:bottom]
        if cropped.shape[0] <= 5 or cropped.shape[1] == 0:
            break
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
            if cropped.shape[0] <= 5 or cropped.shape[1] == 0:
                break
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
    return names[class_index]

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

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def main():
    whites = ["blue","blue","white","white","white","white","white","white","white","white","white","white","white","white","white","white","white","red","red","red","red","red","red","red","red","red","orange","blue","white","blue","blue","red","white","orange","white","blue","orange","orange","orange","red","blue","blue","orange","white","orange","orange","red","blue","orange","red","red","blue","red","white","red","orange","orange","white","white","white","blue"]
    yellows = ["blue", "orange", "blue", "white", "orange", "blue", "white", "orange", "orange", "white", "white", "white", "red", "red", "red", "orange", "orange", "red", "red", "orange", "blue", "orange", "red", "red", "blue", "white", "blue", "blue", "orange", "white", "red", "white", "blue", "blue", "blue", "orange", "blue", "blue", "white", "white", "white", "white", "white", "white", "white", "white", "white", "white", "white", "white", "white", "white", "red", "red", "red", "red", "red", "red", "red", "red", "red"]
    #testing detector

    labCount = {"red":0,"blue":0,"orange":0,"white":0}
    hsvCount = {"red":0,"blue":0,"orange":0,"white":0}
    correctCount = {"red":0,"blue":0,"orange":0,"white":0}

    for i in range(0, 61):
        img = cv2.imread("yellow/piece_" + str(i) + ".jpg")
        color1 = detect_color_lab(img)
        color2 = detect_color_hsv(img)
        color = yellows[i]
        correctCount[color] = correctCount[color] + 1
        if color1 == color:
            labCount[color] = labCount[color] + 1
        if color2 == yellows[i]:
            hsvCount[color] = hsvCount[color] + 1
        print(str(i) + " LAB: " + color1)
        print(str(i) + " HSV: " + color2)
    print("LAB: " + str(labCount))
    print("HSV: " + str(hsvCount))
    print("ACTUAL: " + str(correctCount))

    labs = []
    labs.append(labCount["red"])
    labs.append(labCount["white"])
    labs.append(labCount["orange"])
    labs.append(labCount["blue"])

    corrects = []
    corrects.append(correctCount["red"])
    corrects.append(correctCount["white"])
    corrects.append(correctCount["orange"])
    corrects.append(correctCount["blue"])

    hsvs = []
    hsvs.append(hsvCount["red"])
    hsvs.append(hsvCount["white"])
    hsvs.append(hsvCount["orange"])
    hsvs.append(hsvCount["blue"])

    labels = ['red', 'white', 'orange', 'blue']

    print(labs)
    print(hsvs)
    print(corrects)

    x = np.arange(len(labels))  # the label locations
    width = 0.2 # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, labs, width, label='LAB')
    rects2 = ax.bar(x + width, hsvs, width, label='HSV')
    rects3 = ax.bar(x, corrects, width, label='Actual')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('LAB and HSV classification comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)

    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()