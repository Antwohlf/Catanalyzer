import cv2
import numpy as np
import test_compare_keenan

def gamma_correct(img, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    # nW = int((h * sin) + (w * cos))
    # nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    # M[0, 2] += (nW / 2) - cX
    # M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (w, h))

def detect_dots(img):
    min_threshold = 0 #50                     
    max_threshold = 255 #200                     
    min_area = 8 #400                          
    max_area = 40 #700
    min_circularity = 0.4
    min_inertia_ratio = 0.4

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
    params.minDistBetweenBlobs = 1

    detector = cv2.SimpleBlobDetector_create(params) # create a blob detector object.
    keypoints = detector.detect(img)
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return keypoints, im_with_keypoints

def preprocess(img, gamma):
    grayscale = gamma_correct(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), gamma);
    ret, grayscale = cv2.threshold(grayscale,180,255,cv2.THRESH_TOZERO)
    kernel = np.ones((2,2),np.uint8)
    dilated = cv2.dilate(grayscale,kernel,iterations = 1)
    eroded = cv2.dilate(dilated,kernel,iterations = 1)
    eroded = cv2.dilate(eroded,kernel,iterations = 1)
    return eroded

def rotate_dots(img):
    angles = np.arange(0, 360, 5) #5-degree angle increments
    processed = preprocess(img, 1)
    y_means = []
    for angle in angles:
        rotated = rotate_bound(processed, angle)   
        dots, key_img = detect_dots(rotated)
        y_positions = []
        for dot in dots:
            y_positions.append(dot.pt[1])
        y_means.append(np.mean(y_positions))
    y_max = np.argmax(y_means)
    rotated1 = rotate_bound(processed, angles[y_max])  
    return rotated1, y_max

get_center_color(img):
    

def main():
    nums = np.arange(1, 18)
    img = cv2.imread("coins1/catan_28_65.jpg")
    rotated, y_max = rotate_dots(img)
    edges1 = cv2.Canny(rotated,100,100)
    cv2.imshow("init", edges1)
    cv2.waitKey(0)
    diffs = []
    for num in nums:
        img2 = cv2.imread("coins2/catan_24_" + str(num) + ".jpg")
        rotated2, y_max = rotate_dots(img2)
        edges2 = cv2.Canny(rotated2,100,100)
        compare_image = test_compare_keenan.CompareImage(edges1, edges2)
        image_difference = compare_image.compare_image()
        diffs.append(image_difference)
    
    lowest = np.argmin(diffs)
    matched = cv2.imread("coins2/catan_24_" + str(lowest + 1) + ".jpg")
    matched, y_max = rotate_dots(matched)
    edges2 = cv2.Canny(matched,100,100)
    cv2.imshow("init", edges2)
    cv2.waitKey(0)
    
    print(diffs)
    print(lowest + 1)

if __name__ == "__main__":
    main()