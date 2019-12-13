import numpy as np
import cv2
import sys

def rundarknet(image):
    min_confidence = 0.0
    height,width,ch = image.shape

    #Load names of classes
    classes = None
    with open('../Catanalyzer.names', 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    net = cv2.dnn.readNetFromDarknet('../Catanalyzer.cfg', '../Catanalyzer_final.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (416, 416), True, crop=False)
    net.setInput(blob)
    # Run the preprocessed input blog through the network
    predictions = net.forward()
    probability_index=5

    stuff = []

    for i in range(predictions.shape[0]):
        prob_arr=predictions[i][probability_index:]
        class_index=prob_arr.argmax(axis=0)
        confidence= prob_arr[class_index]
        if confidence > min_confidence:
            x_center=predictions[i][0]*width
            y_center=predictions[i][1]*height
            width_box=predictions[i][2]*width
            height_box=predictions[i][3]*height
            x1=int(x_center-width_box * 0.5)
            y1=int(y_center-height_box * 0.5)
            x2=int(x_center+width_box * 0.5)
            y2=int(y_center+height_box * 0.5)
            #cv2.rectangle(image,(x1,y1),(x2,y2),(255,255,255),1)
            #cv2.putText(image,classes[class_index]+" "+"{0:.1f}".format(confidence),(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
            stuff.append((classes[class_index],x1,y1,x2,y2))
    return stuff

def main():
    image = cv2.imread('../BoardImages/catan_28.jpg',1)
    stuff = rundarknet(image)
    print(stuff)

if __name__ == "__main__":
	main()
