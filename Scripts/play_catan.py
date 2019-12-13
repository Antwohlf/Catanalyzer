import cv2
import numpy as np
import detectColor
from makeOverlay import makeOverlay
from seedice import get_dice_state as get_dice
import streamvideo as stream
from rundarknet import rundarknet
# Dice state machine: The first number that isn't "No Roll"
# Breaks dice into play state. All numbers ignored until next "No Roll"
# -1 means No Roll State
ret, cap = stream.startStream()
stopFeed = False
stopFeed2 = False
dice_states = [(-12, -12)] #No touchy
dice_state = -1 #Start with no roll

while ret and not stopFeed and not stopFeed2:
    ret, frame = cap.read()
    frame = frame[50:frame.shape[0] - 50, 500:frame.shape[1]-200]
    # Frame is our image. All processing happens here

    # ======> Dice State Machine <======
    roll, mode, dice_output, output, show = get_dice(dice_states, frame, kept_states=15, var_thresh=30, gamma=0.7)
    if dice_state == -1 and roll > 0 and mode > 0:
        #Out of No Roll state to Roll state
        dice_state = roll
        print(str(roll))
    elif dice_state >= 2 and roll == -1 and mode < 0:
        #Out of Roll state to No Roll state
        dice_state = -1
        print("No roll")
    
    data = {"dice": dice_state, "red":{"road": 0, "town": 0, "city": 0}, "blue":{"road": 0, "town": 0, "city": 0}, "white":{"road": 0, "town": 0, "city": 0}, "orange":{"road": 0, "town": 0, "city": 0}}

    # =======> Run YOLO/DARKNET <========
    #### TODO
    classes = rundarknet(frame)
    for box in classes:
        label = box[0]
        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        # org 
        org = (int(box[1]), int(box[2]))
        # fontScale 
        fontScale = 0.5
        # Blue color in BGR 
        color = (0, 0, 255) 
        # Line thickness of 2 px 
        thickness = 1
        # Using cv2.putText() method 
        image = cv2.putText(output, label, org, font,  
                        fontScale, color, thickness, cv2.LINE_AA)
        cv2.rectangle(output, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), (255,0,0), 2)

    # =======> Detect Colors <========
    for box in classes:
        cropped = frame[box[1]:box[3], box[2]:box[4]]
        print (box)
        label = box[0]
        if label == "city" or label == "road" or label == "town":
            color = detectColor.detect_color_lab(cropped)
            if color != "unknown":
                    data[color][label] = data[color][label] + 1

    # =======> Detect Coins <========
    #### TODO
    # coins = detectNumber(yolo_bounding_box)

    # =======> Produce Overlay <========
    # fakeData = {"dice": dice_state, "red":{"road":4, "town":2, "city":0}, "white":{"road":3, "town":1, "city":2}, "blue":{"road":3, "town":1, "city":1}, "orange":{"road":5, "town":1, "city":1}}
    overlaid = makeOverlay(cv2.cvtColor(output, cv2.COLOR_BGR2RGBA), data)
    stopFeed2 = stream.displayStream(stream.pil2cv(overlaid), name="Live Catan Feed")

    if(show):
        stopFeed = stream.displayStream(dice_output, name="Live Dice Feed")
    # Resize windows in code to  show both

# Teardown 
stream.endStream(cap)