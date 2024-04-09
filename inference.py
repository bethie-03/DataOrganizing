import cv2
from postprocessing import nms
from configs import *
import os

net = cv2.dnn.readNet(YOLOV4_WEIGHT, YOLOV4_CFGS)
layer_names = net.getLayerNames()
unconnected_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[index - 1] for index in unconnected_indices]

def function_2(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y, drawing, filename, width, height
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            end_x, end_y = x, y
            cv2.rectangle(img_copy, (start_x, start_y), (end_x, end_y), (0, 0, 255), 1)
            cv2.imshow("Detection result", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        if cv2.waitKey(0) & 0xFF == ord('d'):
            cv2.imshow("Detection result", img)
        else:
            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 0, 255), 1)
            
            x_center = (start_x + end_x) / 2
            y_center = (start_y + end_y) / 2
            w = abs(end_x - start_x)
            h = abs(end_y - start_y)

            x_center_norm = x_center / width
            y_center_norm = y_center / height
            w_norm = w / width
            h_norm = h / height
                
            with open(os.path.join(FOLDER_ANNOTATION, f"{filename}.txt"), 'w') as files:
                files.write(f"{0} {x_center_norm} {y_center_norm} {w_norm} {h_norm}\n")
            cv2.imshow("Detection result", img)

drawing = False
start_x, start_y = -1, -1
end_x, end_y = -1, -1

for file in os.listdir(FOLDER_NON_LABEL_IMG):
    filename = file.split('.')[0]
    file_path = os.path.join(FOLDER_NON_LABEL_IMG, file)
    img = cv2.imread(file_path)
    img = cv2.resize(img, None, fx=.4,fy=.4)
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)  
    boxes=[]
    confidences=[]
    boxes_yolo=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            confidence = scores[0]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2
                boxes_yolo.append([detection[0],detection[1],detection[2],detection[3]])
                boxes.append([x, y, x + w, y + h])
                confidences.append(float(confidence))
    picked_boxes = nms(boxes, confidences, NMS_THRESHOLD)
    for idx in picked_boxes:
        x, y, x2, y2 = boxes[idx]
        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 1)
        cv2.putText(img, CLASS[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    cv2.imshow("Detection result", img)
    cv2.setMouseCallback("Detection result", function_2)
    while True:
        if cv2.waitKey(0)==ord('1'):
            with open(os.path.join(FOLDER_ANNOTATION, f"{filename}.txt"), 'a') as files:
                for idx in picked_boxes:
                    center_x,center_y, w, h = boxes_yolo[idx]
                    files.write(f"{0} {center_x} {center_y} {w} {h}\n")
            new_path = os.path.join(FOLDER_LABEL_IMG, file)
            os.rename(file_path, new_path)
            break
    if cv2.waitKey(0)==ord('2'):
        continue
    elif cv2.waitKey(0) == ord('q') or cv2.waitKey(0) == ord('Q'):
            quit()


cv2.destroyAllWindows()