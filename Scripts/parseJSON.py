import json
import numpy as np
from matplotlib.path import Path
import cv2

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
    print(hex)
    return hex

name = "catan_19"
imgPath = "../BoardImages/" + name + ".jpg"
jsonPath = '../BoardAnnotations/' + name + '.json'

roadHex = "#083D77"
cityHex = "#EBEBD3"
townHex = "DA4167"
coinHex = "#F78764"
robberHex = "#F4D35E"

print(imgPath)

origImg = cv2.imread(imgPath)
newImg = np.zeros_like(origImg, dtype=np.uint8)

with open(jsonPath) as json_file:
    data = json.load(json_file)
    regions = data["IMG_20191114_233248.jpg5030684"]["regions"]
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
            cv2.fillConvexPoly(newImg, pts, color)
        elif(shape["name"] == "circle"):
            color = hex2rgb(coinHex)
            if(tags["type"] == "robber"):
                color = hex2rgb(robberHex)
            cv2.circle(newImg,(shape["cx"],shape["cy"]), shape["r"], color, -1)


scale_percent = 25 # percent of original size
width = int(newImg.shape[1] * scale_percent / 100)
height = int(newImg.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(newImg, dim, interpolation = cv2.INTER_AREA)
recolored = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
cv2.imshow("Output", recolored)
cv2.waitKey(0)