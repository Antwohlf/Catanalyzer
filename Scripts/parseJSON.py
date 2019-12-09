import json
import numpy as np
import os
from matplotlib.path import Path
import matplotlib.pyplot as plt
import cv2


# MODIFY THESE
rect_thickness = 2
img_scale = 25 #X% scale on output

NO_FLIPS = [0, 1, 2, 3, 4, 6, 7, \
            9, 10, 11, 18, 19, 20, 21, 22, \
            26, 28, 31, 42, 52, 54, 55, 56, 57]

def point_in_circle(cx, cy, r, x, y):
    a = np.array([cx, cy])
    b = np.array([x, y])
    dist = np.linalg.norm(a-b)
    return dist <= r

def point_in_poly(point, polyCoords):
     return Path(polyCoords).contains_point(point) 

def hex2rgb(h):
    h = h.strip("#")
    hex = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return hex


blackHex = "#000000"
roadHex = "#083D77"
cityHex = "#EBEBD3"
townHex = "DA4167"
coinHex = "#F78764"
robberHex = "#F4D35E"


def bbox_pol(pts):
    xmin = pts[0][0]
    xmax = 0
    ymin = pts[0][1]
    ymax = 0
    
    for pt in pts:
        if pt[0] > xmax:
            xmax = pt[0]
        elif pt[0] < xmin:
            xmin = pt[0]

        if pt[1] > ymax:
            ymax = pt[1]
        elif pt[1] < ymin:
            ymin = pt[1]
    return (xmin, ymin),(xmax, ymax)

def bbox_circ(cx, cy, r):
    xmin = cx - r
    xmax = cx + r
    ymin = cy + r
    ymax = cy - r
    return (xmin, ymax),(xmax, ymin)

def bbox_ellispse(cx, cy, rx, ry):
    xmin = cx - rx
    xmax = cx + rx
    ymin = cy + ry
    ymax = cy - ry
    return (xmin, ymax),(xmax, ymin)

def xml_from_bbox(topLeft, bottomRight):
    left = topLeft[0]
    top = topLeft[1]
    right = bottomRight[0]
    bottom = bottomRight[1]
    w = abs(right - left)
    h = abs(top - bottom)
    cx = left + (w // 2)
    cy = top + (h // 2)
    return cx, cy, w, h

def num_for_label(label):
    if(label == "road"):
        return 0
    elif(label == "town"):
        return 1
    elif(label == "city"):
        return 2
    elif(label == "coin"):
        return 3
    elif(label == "robber"):
        return 4

def draw_bounding(jsonPath, newImg):
    with open(jsonPath) as json_file:
        data = json.load(json_file)
        filename = list(data.keys())[0]
        regions = data[filename]["regions"]
        for index in regions:
            region = regions[index]
            shape = region["shape_attributes"]
            tags = region["region_attributes"]
            if(shape["name"] == "polygon"):
                x = np.array(shape["all_points_x"])
                y = np.array(shape["all_points_y"])
                # Test for inclusion obvi
                pts = np.array(list(zip(x, y)), np.int32)
                color = hex2rgb(roadHex)
                if tags["type"] == "town":
                    color = hex2rgb(townHex)
                elif tags["type"] == "city":
                    color = hex2rgb(cityHex)
                elif(tags["type"] == "robber"):
                    color = hex2rgb(robberHex)
                topLeft, bottomRight = bbox_pol(pts)
                cv2.rectangle(newImg, topLeft, bottomRight, color, rect_thickness)
            elif(shape["name"] == "circle"):
                color = hex2rgb(coinHex)
                if(tags["type"] == "robber"):
                    color = hex2rgb(robberHex)
                topLeft, bottomRight = bbox_circ(shape["cx"], shape["cy"], shape["r"])
                cv2.rectangle(newImg, topLeft, bottomRight, color, rect_thickness)
            elif(shape["name"] == "ellipse"):
                color = hex2rgb(blackHex)
                if(tags["type"] == "coin"):
                    color = hex2rgb(coinHex)
                elif(tags["type"] == "robber"):
                    color = hex2rgb(robberHex)
                topLeft, bottomRight = bbox_ellispse(shape["cx"], shape["cy"], shape["rx"], shape["ry"])
                cv2.rectangle(newImg, topLeft, bottomRight, color, rect_thickness)

def export_bounding(jsonPath, newImg, draw=True):
    #road, town, city, coin, robber
    name = os.path.splitext(jsonPath)[0]
    name = name.split("/")[1]

    # file_no = int(name.split("_")[1])
    # if file_no not in NO_FLIPS:
    #     newImg = np.flip(newImg, axis=(0,1))

    H, W, _ = newImg.shape

    f= open(name + ".txt","w+")
    with open(jsonPath) as json_file:
        data = json.load(json_file)
        filename = list(data.keys())[0]
        regions = data[filename]["regions"]
        for index in regions:
            region = regions[index]
            shape = region["shape_attributes"]
            tags = region["region_attributes"]
            tag = tags["type"]
            if(shape["name"] == "polygon"):
                x = np.array(shape["all_points_x"])
                y = np.array(shape["all_points_y"])
                # Test for inclusion obvi
                pts = np.array(list(zip(x, y)), np.int32)
                topLeft, bottomRight = bbox_pol(pts)
                cx, cy, w, h = xml_from_bbox(topLeft, bottomRight)
                line = str(num_for_label(tag)) + " " + str(cx/W) + " " + str(cy/H) + " " + str(w/W) +" " + str(h/H) + "\n"
                f.write(line)
                if(draw):
                    cv2.circle(newImg,(cx, cy), 8, (0, 0, 0), -1)
            elif(shape["name"] == "circle"):
                topLeft, bottomRight = bbox_circ(shape["cx"], shape["cy"], shape["r"])
                cx, cy, w, h = xml_from_bbox(topLeft, bottomRight)
                line = str(num_for_label(tag)) + " " + str(cx/W) + " " + str(cy/H) + " " + str(w/W) +" " + str(h/H) + "\n"
                f.write(line)
                if(draw):
                    cv2.circle(newImg,(cx, cy), 8, (0, 0, 0), -1)
            elif(shape["name"] == "ellipse"):
                topLeft, bottomRight = bbox_ellispse(shape["cx"], shape["cy"], shape["rx"], shape["ry"])
                cx, cy, w, h = xml_from_bbox(topLeft, bottomRight)
                line = str(num_for_label(tag)) + " " + str(cx/W) + " " + str(cy/H) + " " + str(w/W) + " " + str(h/H) + "\n"
                f.write(line)
                if(draw):
                    cv2.circle(newImg,(cx, cy), 8, (0, 0, 0), -1)
        f.close() 

def create_semantic(jsonPath, newImg):
    with open(jsonPath) as json_file:
        data = json.load(json_file)
        filename = list(data.keys())[0]
        regions = data[filename]["regions"]

        for index in regions:
            region = regions[index]
            shape = region["shape_attributes"]
            tags = region["region_attributes"]
            if(shape["name"] == "polygon"):
                x = np.array(shape["all_points_x"])
                y = np.array(shape["all_points_y"])
                # Test for inclusion obvi
                pts = np.array(list(zip(x, y)), np.int32)
                color = hex2rgb(blackHex)
                if tags["type"] == "road":
                    color = hex2rgb(roadHex)
                elif tags["type"] == "town":
                    color = hex2rgb(townHex)
                elif tags["type"] == "city":
                    color = hex2rgb(cityHex)
                elif(tags["type"] == "robber"):
                    color = hex2rgb(robberHex)
                elif(tags["type"] == "coin"):
                    color = hex2rgb(coinHex)
                cv2.fillConvexPoly(newImg, pts, color)

            elif(shape["name"] == "circle"):
                color = hex2rgb(blackHex)
                if(tags["type"] == "coin"):
                    color = hex2rgb(coinHex)
                elif(tags["type"] == "robber"):
                    color = hex2rgb(robberHex)
                cv2.circle(newImg,(shape["cx"],shape["cy"]), shape["r"], color, -1)

            elif(shape["name"] == "ellipse"):
                color = hex2rgb(blackHex)
                if(tags["type"] == "coin"):
                    color = hex2rgb(coinHex)
                elif(tags["type"] == "robber"):
                    color = hex2rgb(robberHex)

                cx = int(shape["cx"])
                cy = int(shape["cy"])
                majorAxis = int(shape["rx"])
                minorAxis = int(shape["ry"])
                if majorAxis < minorAxis:
                    majorAxis = int(shape["ry"])
                    minorAxis = int(shape["rx"])
                cv2.ellipse(newImg, (cx, cy), (minorAxis, majorAxis), 90.0, 0.0, 360.0, color, -1)

def scale_and_show(newImg):
    scale_percent = img_scale # percent of original size
    width = int(newImg.shape[1] * scale_percent / 100)
    height = int(newImg.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(newImg, dim, interpolation = cv2.INTER_AREA)
    recolored = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    plt.imshow(recolored)
    plt.show()

def main():
    image_names = os.listdir("BoardImages")
    json_names = os.listdir("BoardAnnotations")

    # sanity check, check for equality
    assert(len(image_names) == len(json_names))

    for i in range(len(image_names)):
        print("file: {:s}".format(image_names[i]))
        img_path = os.path.join("BoardImages", image_names[i])
        json_path = os.path.join("BoardAnnotations", json_names[i])

        orig_img = cv2.imread(img_path)
        new_img = np.zeros_like(orig_img, dtype=np.uint8)

        create_semantic(json_path, new_img)
        # draw_bounding(json_path, new_img)
        export_bounding(json_path, new_img, True)
        
        # scale_and_show(new_img)
    # name = "catan_1"
    # imgPath = "../BoardImages/" + name + ".jpg"
    # jsonPath = '../BoardAnnotations/' + name + '.json'
    # origImg = cv2.imread(imgPath)
    # newImg = np.zeros_like(origImg, dtype=np.uint8)

if __name__ == "__main__":
    main()


