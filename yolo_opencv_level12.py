#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import argparse
import numpy as np
import os
import tensorflow as tf
import math
from numpy import zeros

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--folder', required=True,
                help = 'path to the folder need to be detected')
# ap.add_argument('-i', '--image', required=True,
#                 help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_target(img, class_id, x, y, x_plus_w, y_plus_h):

    label = str(class_id)

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (0,0,255), 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def getint(name):
    num = name.split('.')[0]
    num = num[5:]
    return int(num)

def drawPersonBoxInFrame():
    global classes, COLORS

    dirs = os.listdir(args.folder)
    dirs.sort(key=getint)

    classes = None

    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    for dir in dirs:
        image_path = args.folder + "/" + dir
        image = cv2.imread(image_path)

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        classes = None

        with open(args.classes, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        net = cv2.dnn.readNet(args.weights, args.config)

        blob = cv2.dnn.blobFromImage(image, scale, (608,608), (0,0,0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4


        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])


        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            if class_ids[i] == 0:
                print(image_path + " class: " + str(class_ids[i]) + " x: " + str(x) + " y: " + str(y) + " w: " + str(w) + " h: " + str(h))
                draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
            
        cv2.imwrite(args.folder + "_detect/"+dir, image)

def yolo_detect(image):
    global classes, COLORS
    global class_ids
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(args.weights, args.config)

    blob = cv2.dnn.blobFromImage(image, scale, (608,608), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return indices, boxes

def personBoxInfo():
    global classes, COLORS
    global target_x, target_y, target_w, target_h, target_id

    dirs = os.listdir(args.folder)
    dirs.sort(key=getint)
    for i,dir in enumerate(dirs):
        print([target_id, target_x,target_y,target_w,target_h])
        image_cur_path = args.folder + "/" + dirs[i]
        image = cv2.imread(image_cur_path)
        if (i == 0):
            indices, boxes = yolo_detect(cv2.imread(image_cur_path))
            distance_gap = 9999
            size_gap = 9999
            low_score = 0
            print("-------------------------------------")
            for i in indices:
                box = boxes[i]
                x = int(box[0])
                y = int(box[1])
                w = int(box[2])
                h = int(box[3])
                if class_ids[i] == 0:
                    print(image_cur_path + " class: " + str(class_ids[i]) + " x: " + str(x) + " y: " + str(y) + " w: " + str(w) + " h: " + str(h))
                    current_distance_gap = math.sqrt((target_x - x)**2 + (target_y - y)**2)
                    current_size_gap = ((target_w - w) + (target_h - h))
                    print(image_cur_path + " class: " + str(class_ids[i]) + " distance: " + str(current_distance_gap) + " area: " + str(current_size_gap))
                    if current_distance_gap < distance_gap and current_size_gap < size_gap:
                        distance_gap = current_distance_gap
                        size_gap = current_size_gap
                        target_box = box
                        # print("get candidate")
            
            target_box[0] = int(target_box[0])
            target_box[1] = int(target_box[1])
            target_box[2] = int(target_box[2])
            target_box[3] = int(target_box[3])
            print("-------------------------------------")
            print(image_cur_path + " Target " + " x: " + str(target_box[0]) + " y: " + str(target_box[1]) + " w: " + str(target_box[2]) + " h: " + str(target_box[3]))
            print("-------------------------------------")
            draw_target(image, target_id, round(target_box[0]), round(target_box[1]), round(target_box[0]+target_box[2]), round(target_box[1]+target_box[3]))
            target_x = target_box[0]
            target_y = target_box[1]
            target_w = target_box[2]
            target_h = target_box[3]
            cv2.imwrite(args.folder + "_track/"+dir, image)
        
        image_cur_path = args.folder + "/" + dirs[i]
        image_pre_path = args.folder + "/" + dirs[i-1]

        
        indices, boxes = yolo_detect(cv2.imread(image_cur_path))

        # ref_indices, ref_boxes = yolo_detect(cv2.imread(image_cur_path))
        # next_indices, next_boxes = yolo_detect(cv2.imread(image_pre_path))

        distance_gap = 9999
        size_gap = 9999
        low_score = 0
        print("-------------------------------------")
        for i in indices:
            box = boxes[i]
            x = int(box[0])
            y = int(box[1])
            w = int(box[2])
            h = int(box[3])
            if class_ids[i] == 0:
                print(image_cur_path + " class: " + str(class_ids[i]) + " x: " + str(x) + " y: " + str(y) + " w: " + str(w) + " h: " + str(h))
                current_distance_gap = math.sqrt((target_x - x)**2 + (target_y - y)**2)
                current_size_gap = ((target_w - w) + (target_h - h))

                if current_distance_gap < distance_gap and current_size_gap < size_gap:
                    distance_gap = current_distance_gap
                    size_gap = current_size_gap
                    target_box = box
                    # print("get candidate")
        
        target_box[0] = int(target_box[0])
        target_box[1] = int(target_box[1])
        target_box[2] = int(target_box[2])
        target_box[3] = int(target_box[3])
        print("-------------------------------------")
        print(image_cur_path + " Target " + " x: " + str(target_box[0]) + " y: " + str(target_box[1]) + " w: " + str(target_box[2]) + " h: " + str(target_box[3]))
        print("-------------------------------------")
        draw_target(image, target_id, round(target_box[0]), round(target_box[1]), round(target_box[0]+target_box[2]), round(target_box[1]+target_box[3]))
        target_x = target_box[0]
        target_y = target_box[1]
        target_w = target_box[2]
        target_h = target_box[3]
        cv2.imwrite(args.folder + "_track/"+dir, image)
        

def getTarget():
    global target_x, target_y, target_w, target_h, target_id
    ground_truth = args.folder + ".txt"
    with open(ground_truth, 'r') as f:
        target = f.readlines()[0]
        target_id = int(target.split(',')[1])
        target_x = int(target.split(',')[2])
        target_y = int(target.split(',')[3])
        target_w = int(target.split(',')[4])
        target_h = int(target.split(',')[5])
    # print([target_id, target_x,target_y,target_w,target_h])
    

def main():
    # drawPersonBoxInFrame()
    getTarget()
    personBoxInfo()
    

main()
# input("Press Enter to continue...")
