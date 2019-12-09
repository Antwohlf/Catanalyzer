import json
import numpy as np
import os

import matplotlib.pyplot as plt

from shutil import rmtree

import cv2


INSTANCES = {'road': 0, 'coin': 0, 'robber': 0, 'city': 0, 'town': 0}

NO_FLIPS = [0, 1, 2, 3, 4, 6, 7, \
            9, 10, 11, 18, 19, 20, 21, 22, \
            26, 28, 31, 42, 52, 54, 55, 56, 57]

TAGS = {'road': 0, 'coin': 1, 'robber': 2, 'city': 3, 'town': 4}

def bbox_pol(pts):
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    return (xmin, ymin),(xmax, ymax)

def bbox_circ(cx, cy, r):
    xmin = cx - r
    xmax = cx + r
    ymin = cy + r
    ymax = cy - r

    if (ymax < 0): ymax = 0
    if (xmin < 0): xmin = 0

    return (xmin, ymax),(xmax, ymin)

def bbox_ellispse(cx, cy, rx, ry):
    xmin = cx - rx
    xmax = cx + rx
    ymin = cy + ry
    ymax = cy - ry

    if (ymax < 0): ymax = 0
    if (xmin < 0): xmin = 0

    return (xmin, ymax),(xmax, ymin)

def read_json(jsonfile):
    with open(jsonfile, "r") as f_json:
        data = json.load(f_json)
    filename = list(data.keys())[0]
    return data[filename]["regions"]

def yolo_from_bbox(topLeft, bottomRight):
    left = topLeft[0]
    top = topLeft[1]
    right = bottomRight[0]
    bottom = bottomRight[1]
    w = abs(right - left)
    h = abs(top - bottom)
    cx = left + (w // 2)
    cy = top + (h // 2)
    return cx, cy, w, h

def parse_regions(region_json, corresponding_image, file_no):
    if file_no not in NO_FLIPS:
        corresponding_image = np.flip(corresponding_image)
    for index in region_json:
        region = region_json[index]
        shape = region["shape_attributes"]
        tags = region["region_attributes"]
        tag = tags["type"]

        if shape["name"] == "polygon":
            x = np.array(shape["all_points_x"])
            y = np.array(shape["all_points_y"])
            # Test for inclusion obvi
            pts = np.array(list(zip(x, y)), np.int32)
            topLeft, bottomRight = bbox_pol(pts)
        elif shape["name"] == "circle":
            topLeft, bottomRight = bbox_circ(shape["cx"], shape["cy"], shape["r"])
        elif shape["name"] == "ellipse":
            topLeft, bottomRight = bbox_ellispse(shape["cx"], shape["cy"], shape["rx"], shape["ry"])

        H, W, _ = np.shape(corresponding_image)

        image_crop = corresponding_image[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]
        if file_no not in NO_FLIPS:
            image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)

        cx, cy, w, h = yolo_from_bbox(topLeft, bottomRight)

            
        yolo_string = str(TAGS[tag]) + ' ' + str(cx) + " " + str(cy) + " " + str(w) + " " + str(h)
        with open("dataset/{:s}/pos_{:d}.txt".format(tag, INSTANCES[tag]), "w+") as f_write:
            f_write.write(yolo_string)
        cv2.imwrite("dataset/{:s}/pos_{:d}.jpg".format(tag, INSTANCES[tag]), image_crop)
        INSTANCES[tag] += 1

def main():
    print("i parse all the things and make all the labelled objects")

    if os.path.isdir("dataset"):
        rmtree("dataset")
    
    os.makedirs("dataset")

    for obj_class in ["road", "town", "city", "coin", "robber"]:
        os.makedirs("dataset/{:s}".format(obj_class))

    image_names = os.listdir("BoardImages")
    json_names = os.listdir("BoardAnnotations")

    # sanity check, check for equality
    assert(len(image_names) == len(json_names))

    for i in range(len(image_names)):
        print("processing the {:d}th image".format(i))
        json_fileno = int(json_names[i].split('.')[0].split('_')[1])
        image_fileno =int(image_names[i].split('.')[0].split('_')[1])
        assert(json_fileno == image_fileno)
        data_regions = read_json(os.path.join("BoardAnnotations", json_names[i]))
        parse_regions(data_regions, cv2.imread(os.path.join("BoardImages", image_names[i])), json_fileno)

if __name__ == "__main__":
    main()

