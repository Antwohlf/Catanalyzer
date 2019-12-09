import os
import cv2

image_names = os.listdir("../BoardImages")

for i in range(len(image_names)):
    if "png" in image_names[i] or "jpg" in image_names[i]:
        print("file: {:s}".format(image_names[i]))
        img_path = os.path.join("../BoardImages", image_names[i])
        orig_img = cv2.imread(img_path)
        grayscale = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        new_path = os.path.join("../BoardImages/grayscale/", "gray_" + image_names[i] + ".png")
        cv2.imwrite(new_path, grayscale)