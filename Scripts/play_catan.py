import cv2
import numpy as np
from seedice import get_dice_state as get_dice
import streamvideo as stream

def main():
    ret, cap = stream.startStream()
    stopFeed = False

    dice_states = []

    while ret and not stopFeed:
        ret, frame = cap.read()
        # Frame is our image. All processing happens here
        roll, output = get_dice(dice_states, frame)
        stopFeed = stream.displayStream(output)
    # Teardown 
    stream.endStream(cap)

if __name__ == "__main__":
    main()