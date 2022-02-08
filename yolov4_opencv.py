# pip install --upgrade opencv-python
# pip install wget

import wget
import cv2
import numpy as np

# yolo_cfg = wget.download('https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.cfg')
# yolo_weights = wget.download('https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights')
# coco_names = wget.download('https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image1 = cv2.imread('Gare-Saint-Lazare.jpg')

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    image = image.copy()
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)
 
cv2.imshow('image originale', ResizeWithAspectRatio(image1, width=700))

with open('coco.names', 'rt') as f:
    labels = f.read().rstrip('\n').split('\n')

np.random.seed(45)
BOX_COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

def analyze_photo(image):
  (H, W) = image.shape[:2]
  yolo = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
  yololayers = [yolo.getLayerNames()[i - 1] for i in yolo.getUnconnectedOutLayers()]
  blobimage = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
  yolo.setInput(blobimage)

  layerOutputs = yolo.forward(yololayers)

  boxes_detected = []
  confidences_scores = []
  labels_detected = []
  # loop over each of the layer outputs
  for output in layerOutputs:
    # loop over each of the detections
    for detection in output:
      # extract the class ID and confidence (i.e., probability) of the current object detection
      scores = detection[5:]
      classID = np.argmax(scores)
      confidence = scores[classID]
  
      # Take only predictions with confidence more than CONFIDENCE_MIN thresold
      if confidence > 0.5:
        # Bounding box
        box = detection[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = box.astype("int")
  
        # Use the center (x, y)-coordinates to derive the top and left corner of the bounding box
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))
  
        # update our result list (detection)
        boxes_detected.append([x, y, int(width), int(height)])
        confidences_scores.append(float(confidence))
        labels_detected.append(classID)

  final_boxes = cv2.dnn.NMSBoxes(boxes_detected, confidences_scores, 0.5, 0.5)

  image_finale = image.copy()
  # loop through the final set of detections remaining after NMS and draw bounding box and write text
  for max_valueid in final_boxes:
      max_class_id = max_valueid
  
      # extract the bounding box coordinates
      (x, y) = (boxes_detected[max_class_id][0], boxes_detected[max_class_id][1])
      (w, h) = (boxes_detected[max_class_id][2], boxes_detected[max_class_id][3])
  
      # draw a bounding box rectangle and label on the image
      color = [int(c) for c in BOX_COLORS[labels_detected[max_class_id]]]
      cv2.rectangle(image_finale, (x, y), (x + w, y + h), color, 1)
      
      score = str(round(float(confidences_scores[max_class_id]) * 100, 1)) + "%"
      text = "{}: {}".format(labels[labels_detected[max_class_id]], score)
      cv2.putText(image_finale, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
       
  return image_finale

img1 = analyze_photo(image1)
cv2.imshow('image finale', ResizeWithAspectRatio(img1, width=700))

cv2.waitKey(0) 
cv2.destroyAllWindows()