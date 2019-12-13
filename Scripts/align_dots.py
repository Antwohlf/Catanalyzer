import cv2
import numpy as np
import test_compare_keenan
import calcNumber

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
    min_area = 10 #400                          
    max_area = 120 #700
    min_circularity = 0.5
    min_inertia_ratio = 0.5

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
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.GaussianBlur(grayscale,(5,5),cv2.BORDER_DEFAULT)
    ret, grayscale = cv2.threshold(grayscale,150,220,cv2.THRESH_BINARY)
    color = get_center_color(img)
    kernel = np.ones((7,7),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)
    eroded = cv2.dilate(grayscale,kernel,iterations = 1)
    dilated = cv2.erode(eroded,kernel2,iterations = 1)
    return grayscale


def rotate_dots(img):
    processed = preprocess(img, 1)
    angles = np.arange(0, 280, 5) #5-degree angle increments
    y_means = []
    y_var = []

    count = 0 
    skip = True

    for angle in angles:
        rotated = rotate_bound(processed, angle)   
        dots, key_img = detect_dots(rotated)
        cv2.imshow("die", key_img)
        cv2.waitKey(0)
        y_positions = []
        for dot in dots:
            y_positions.append(dot.pt[1])
        if len(dots) > 0:
            var = np.var(y_positions)
            mean = np.mean(y_positions)
            y_means.append(mean)
            y_var.append(var)

        #Leave early if we are progessing upwards
        if count is 4:
            if (len(y_var) > 0 and y_var[0] <= var) and (len(y_means) > 0 and y_means[0] > mean):
                skip = False
                break
        count = count + 1
    if not skip:
        angles = np.arange(360, 280, -5) #5-degree angle increments
        y_means = []
        y_var = []

        for angle in angles:
            rotated = rotate_bound(processed, angle)   
            dots, key_img = detect_dots(rotated)
            cv2.imshow("die", key_img)
            cv2.waitKey(0)
            y_positions = []
            for dot in dots:
                y_positions.append(dot.pt[1])
            var = np.var(y_positions)
            mean = np.mean(y_positions)
            y_means.append(mean)
            y_var.append(var)
    if len(y_means) > 0 and len(y_var) > 0:
        y_mean = np.argmax(y_means)
        var_min = np.argmin(y_var)
        rotated1 = rotate_bound(processed, angles[y_mean])
        return rotated1, y_mean
    else:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 10000000

def get_center_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0,100,80])
    upper_red = np.array([10,255,230])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = crop_inward(mask, 30, 30)

    if np.mean(mask) > 5:
        return "red"
    else:
        lower_red2 = np.array([80,100,80])
        upper_red2 = np.array([180,255,230])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        if np.mean(mask2) > 3:
            return "red"
    return "black"

def crop_inward(img, crop_amt_x, crop_amt_y):
    crop_img = img[crop_amt_x:-crop_amt_x, crop_amt_y:-crop_amt_y]
    return crop_img

def find_match(img):
    #preprocess
    rotated, y_max = rotate_dots(img)
    rotated = crop_inward(rotated, 20, 30)
    color = get_center_color(img)
    diffs = []
    
    nums = [2,3,4,5,6,8,9,10,11,12]

    for num in nums:

        img2 = cv2.imread("../BoardCoins/" + str(num) + ".jpg")
        thisColor = get_center_color(img2)
        
        if thisColor is color:
            rotated2, y_max = rotate_dots(img2)
            rotated2 = crop_inward(rotated2, 10, 30)
            compare_image = test_compare_keenan.CompareImage(img, img2, 1)
            image_difference = compare_image.compare_image()
            diffs.append(image_difference)
        else:
            diffs.append(100000)
    
    lowest = np.argmin(diffs)
    return nums[lowest]

def find_match_ant(img):
    #preprocess
    cv2.imshow("die", img)
    cv2.waitKey(0)
    rotated, y_max = rotate_dots(img)
    rotated = crop_inward(rotated, 20, 30)
    color = get_center_color(img)
    print(color)
    diffs = []
    
    nums = [2,3,4,5,6,8,9,10,11,12]

    for num in nums:

        img2 = cv2.imread("../BoardCoins/" + str(num) + ".jpg")
        thisColor = get_center_color(img2)
        
        if thisColor is color:
            rotated2, y_max = rotate_dots(img2)
            rotated2 = crop_inward(rotated2, 10, 30)
            image_difference = calcNumber.mse(rotated, rotated2)
            diffs.append(image_difference)
        else:
            diffs.append(100000)
    print(diffs)
    lowest = np.argmin(diffs)
    return nums[lowest]

def find_match_norot_ant(img):
    #preprocess
    # rotated, y_max = rotate_dots(img)
    # rotated = crop_inward(rotated, 20, 30)
    # color = get_center_color(img)
    diffs = []
    
    nums = [2,3,4,5,6,8,9,10,11,12]

    for num in nums:

        img2 = cv2.imread("../BoardCoins/" + str(num) + ".jpg")
        # thisColor = get_center_color(img2)
        
        # if thisColor is color:
            # rotated2, y_max = rotate_dots(img2)
            # rotated2 = crop_inward(rotated2, 10, 30)
        image_difference = calcNumber.mse(img, img2)
        diffs.append(image_difference)
        # else:
        # diffs.append(10000)
    lowest = np.argmin(diffs)
    return nums[lowest]

def find_match_norot(img):
    #preprocess
    # rotated, y_max = rotate_dots(img)
    # rotated = crop_inward(rotated, 20, 30)
    # color = get_center_color(img)
    diffs = []
    
    nums = [2,3,4,5,6,8,9,10,11,12]

    for num in nums:

        img2 = cv2.imread("../BoardCoins/" + str(num) + ".jpg")
        # thisColor = get_center_color(img2)
        
        # if thisColor is color:
            # rotated2, y_max = rotate_dots(img2)
            # rotated2 = crop_inward(rotated2, 10, 30)
        compare_image = test_compare_keenan.CompareImage(img, img2, 1)
        image_difference = compare_image.compare_image()
        diffs.append(image_difference)
        # else:
        # diffs.append(10000)
    lowest = np.argmin(diffs)
    return nums[lowest]
    

def main():
    # process coin
    coin_folder = "coins1"
    catan = 28
    coins_start = 65

    no_rot = []
    rot = []
    rot_ant = []
    no_rot_ant = []
    rand = []
    actuals = [6,2,11,4,12,11,10,8,6,9,3,5,5,10,4,6,8,3]
    actuals2 = [6,8,10,11,12,11,6,4,2,6,4,10,5,6,3,5,3,8]
    count = 0 
    for coin in range(coins_start, coins_start + len(actuals)):
        print(coin)
        img = cv2.imread("coins1/catan_" + str(catan) + "_" + str(coin) + ".jpg")
        rot.append(find_match(img))
        no_rot.append(find_match_norot(img))
        rot_ant.append(find_match_ant(img))
        no_rot_ant.append(find_match_norot_ant(img))
        rand.append(np.random.randint(12, size=1)[0])
        print(str(actuals[count]) + ":" + str(rot[count]) + ":" + str(no_rot[count]) + ":" + str(rot_ant[count]) + ":" + str(no_rot_ant[count]))
        count += 1

    no_rot_ct = 0
    rot_ct = 0
    no_rot_ant_ct = 0
    rot_ant_ct = 0
    rand_ct = 0
    
    for i in range(0, len(actuals)):
        no_rot_ct += no_rot[i] == actuals[i]
        rot_ct += rot[i] == actuals[i]
        no_rot_ant_ct += no_rot_ant[i] == actuals[i]
        rot_ant_ct += rot_ant[i] == actuals[i]
        rand_ct += rand[i] == actuals[i]

    print(no_rot_ct)
    print(rot_ct)
    print(no_rot_ant_ct)
    print(rot_ant_ct)
    print(rand_ct)

if __name__ == "__main__":
    main()