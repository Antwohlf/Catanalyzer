import cv2
import numpy as np
# from colorDetector import detect_color_lab
from makeOverlay import makeOverlay
from seedice import get_dice_state as get_dice
import streamvideo as stream

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
    
    # =======> Run YOLO/DARKNET <========
    #### TODO
    # classes = runYolo(frame)

    # =======> Detect Colors <========
    #### TODO
    # colors = detect_color_lab(yolo_bounding_box)

    # =======> Detect Coins <========
    #### TODO
    # coins = detectNumber(yolo_bounding_box)

    # =======> Produce Overlay <========
    fakeData = {"dice": dice_state, "red":{"road":4, "town":2, "city":0}, "white":{"road":3, "town":1, "city":2}, "blue":{"road":3, "town":1, "city":1}, "orange":{"road":5, "town":1, "city":1}}
    overlaid = makeOverlay(cv2.cvtColor(output, cv2.COLOR_BGR2RGBA), fakeData)
    stopFeed2 = stream.displayStream(stream.pil2cv(overlaid), name="Live Catan Feed")
    if(show):
        stopFeed = stream.displayStream(dice_output, name="Live Dice Feed")
    # Resize windows in code to  show both

# Teardown 
stream.endStream(cap)