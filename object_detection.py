import cv2
import numpy as np

#read yolo pretrained weights and configuration file
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")


#Using coco text file for identifying class name and append in the classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.rstrip() for line in f.readlines()]



layers_name = net.getLayerNames()    #get all the layers inside yolo 

#getting the output layers for defining what object is detected
output_layers =[layers_name[i[0]-1] for i in net.getUnconnectedOutLayers() ]


#capture the video using your computer 
cap = cv2.VideoCapture(0)

while True:
    ret,img = cap.read()
    img = cv2.resize(img,None,fx=0.8,fy=0.6)
    height, width, channels = img.shape
    

    #getting the blob from the images
    blob = cv2.dnn.blobFromImage(img,0.00392,(512,512),(0,0,0),True,crop=False)

    #passing the blob from images to yolo pretrained weights and configuration file 
    net.setInput(blob)
    outs = net.forward(output_layers)


    class_ids = []
    confidences = []
    boxes = []

    #defining the confidence of the object
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence>0.5:
                centre_x = int(detection[0]*width)
                centre_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                # cv2.circle(img,(centre_x,centre_y),10,(0,255,0),2)
                x = int(centre_x-w/2)
                y = int(centre_y-h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)



    #putting the bounding box inside the object
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)
            cv2.putText(img,label,(x,y+30),font,1,(0,255,0),2)


    #showing the predicted objects
    cv2.imshow('Detection video',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





