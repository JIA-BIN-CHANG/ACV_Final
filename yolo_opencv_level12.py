import cv2
import argparse
import numpy as np
import os
import tensorflow as tf
import math
from numpy import zeros
import time

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--folder', required=True,
                help = 'path to the folder need to be detected')
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


def draw_target(img, target_list):
    for id in range(len(target_list)):
        label = target_list[id][0]
        label = str(label)

        x = target_list[id][1]
        y = target_list[id][2]
        w = target_list[id][3]
        h = target_list[id][4]

        # color = COLORS[label]
        if (id == 0):
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)

            cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)



def getint(name):
    num = name.split('.')[0]
    num = num[5:]
    return int(num)

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

def level1():
    path = args.folder+"_result.txt"
    f = open(path, 'w')
    target_list = [[20,1723,274,202,553]]
    # print(target_list)
    dirs = os.listdir(args.folder)
    dirs.sort(key=getint)
    for dir_num,dir in enumerate(dirs[:]):
        image_cur_path = args.folder + "/" + dirs[dir_num]
        image = cv2.imread(image_cur_path)
        indices, boxes = yolo_detect(cv2.imread(image_cur_path))
        # print(len(indices))
        for id in range(len(target_list)):
            # target_id = target_list[id][0]
            target_x = target_list[id][1]
            target_y = target_list[id][2]
            target_w = target_list[id][3]
            target_h = target_list[id][4]
            if (dir_num == 0):
                distance_gap = 9999
                size_gap = 9999
                # print("******************************************************")
                for i in indices:
                    box = boxes[i]
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2])
                    h = int(box[3])
                    if class_ids[i] == 0:
                        # print(image_cur_path + " class: " + str(class_ids[i]) + " x: " + str(x) + " y: " + str(y) + " w: " + str(w) + " h: " + str(h))
                        current_distance_gap = math.sqrt(((target_x+0.5*target_w) - (x+0.5*w))**2 + ((target_y+0.5*target_h) - (y+0.5*h))**2)
                        current_size_gap = abs((target_w - w) + (target_h - h))
                        # print(image_cur_path + " class: " + str(class_ids[i]) + " distance: " + str(current_distance_gap) + " area: " + str(current_size_gap))
                        if current_distance_gap < distance_gap and current_size_gap < size_gap:
                            distance_gap = current_distance_gap
                            size_gap = current_size_gap
                            target_box = box
                            # print("get candidate")
                # print("******************************************************")
                target_list[id][1] = int(target_box[0])
                target_list[id][2] = int(target_box[1])
                target_list[id][3] = int(target_box[2])
                target_list[id][4] = int(target_box[3])
                # print(image_cur_path + " Target "+ str(id) + " x: " + str(target_box[0]) + " y: " + str(target_box[1]) + " w: " + str(target_box[2]) + " h: " + str(target_box[3]))  

            else:

                image_cur_path = args.folder + "/" + dirs[dir_num]
                image = cv2.imread(image_cur_path)
            
                distance_gap = 9999
                size_gap = 9999
                get_candidate = False
                # print("******************************************************")
                for i in indices:
                    box = boxes[i]
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2])
                    h = int(box[3])
                    if class_ids[i] == 0:
                        # print(image_cur_path + " class: " + str(class_ids[i]) + " x: " + str(x) + " y: " + str(y) + " w: " + str(w) + " h: " + str(h))
                        current_distance_gap = math.sqrt(((target_x+0.5*target_w) - (x+0.5*w))**2 + ((target_y+0.5*target_h) - (y+0.5*h))**2)
                        current_size_gap = abs((target_w - w) + (target_h - h))
                        # print(image_cur_path + " class: " + str(class_ids[i]) + " distance: " + str(current_distance_gap) + " area: " + str(current_size_gap))
                        if current_distance_gap < distance_gap and current_size_gap < size_gap:
                                distance_gap = current_distance_gap
                                size_gap = current_size_gap
                                target_box = box
                                get_candidate = True


                # print("******************************************************")
                if (get_candidate == True):
                    target_list[id][1] = int(target_box[0])
                    target_list[id][2] = int(target_box[1])
                    target_list[id][3] = int(target_box[2])
                    target_list[id][4] = int(target_box[3])

                # print(image_cur_path + " Target "+ str(id) + " x: " + str(target_box[0]) + " y: " + str(target_box[1]) + " w: " + str(target_box[2]) + " h: " + str(target_box[3]))  

        print(target_list)
        for target in target_list:
            f.write(str(dirs[dir_num].split('.')[0][5:]))
            f.write(', ')
            for info in target[:-1]:
                f.write(str(info))
                f.write(', ')
            f.write(str(target[-1]))
            f.write('\n')
        draw_target(image, target_list)
        cv2.imwrite(args.folder + "_track/"+dir, image)
    f.close()

def level2():
    path = args.folder+"_result.txt"
    f = open(path, 'w')
    target_list = [[20,1723,274,202,553]]
    # print(target_list)
    dirs = os.listdir(args.folder)
    dirs.sort(key=getint)
    for dir_num,dir in enumerate(dirs[:]):
        image_cur_path = args.folder + "/" + dirs[dir_num]
        image = cv2.imread(image_cur_path)
        indices, boxes = yolo_detect(cv2.imread(image_cur_path))
        # print(len(indices))
        for id in range(len(target_list)):
            # target_id = target_list[id][0]
            target_x = target_list[id][1]
            target_y = target_list[id][2]
            target_w = target_list[id][3]
            target_h = target_list[id][4]
            if (dir_num == 0):
                distance_gap = 9999
                size_gap = 9999
                # print("******************************************************")
                for i in indices:
                    box = boxes[i]
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2])
                    h = int(box[3])
                    if class_ids[i] == 0:
                        # print(image_cur_path + " class: " + str(class_ids[i]) + " x: " + str(x) + " y: " + str(y) + " w: " + str(w) + " h: " + str(h))
                        current_distance_gap = math.sqrt(((target_x+0.5*target_w) - (x+0.5*w))**2 + ((target_y+0.5*target_h) - (y+0.5*h))**2)
                        current_size_gap = abs((target_w - w) + (target_h - h))
                        # print(image_cur_path + " class: " + str(class_ids[i]) + " distance: " + str(current_distance_gap) + " area: " + str(current_size_gap))
                        if current_distance_gap < distance_gap and current_size_gap < size_gap:
                            distance_gap = current_distance_gap
                            size_gap = current_size_gap
                            target_box = box
                            # print("get candidate")
                # print("******************************************************")
                target_list[id][1] = int(target_box[0])
                target_list[id][2] = int(target_box[1])
                target_list[id][3] = int(target_box[2])
                target_list[id][4] = int(target_box[3])
                # print(image_cur_path + " Target "+ str(id) + " x: " + str(target_box[0]) + " y: " + str(target_box[1]) + " w: " + str(target_box[2]) + " h: " + str(target_box[3]))  

            else:

                image_cur_path = args.folder + "/" + dirs[dir_num]
                image = cv2.imread(image_cur_path)
            
                distance_gap = 9999
                size_gap = 9999
                get_candidate = False
                # print("******************************************************")
                for i in indices:
                    box = boxes[i]
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2])
                    h = int(box[3])
                    if class_ids[i] == 0:
                        # print(image_cur_path + " class: " + str(class_ids[i]) + " x: " + str(x) + " y: " + str(y) + " w: " + str(w) + " h: " + str(h))
                        current_distance_gap = math.sqrt(((target_x+0.5*target_w) - (x+0.5*w))**2 + ((target_y+0.5*target_h) - (y+0.5*h))**2)
                        current_size_gap = abs((target_w - w) + (target_h - h))
                        # print(image_cur_path + " class: " + str(class_ids[i]) + " distance: " + str(current_distance_gap) + " area: " + str(current_size_gap))
                        if current_distance_gap < distance_gap and current_size_gap < size_gap:
                                distance_gap = current_distance_gap
                                size_gap = current_size_gap
                                target_box = box
                                get_candidate = True


                # print("******************************************************")
                if (get_candidate == True):
                    target_list[id][1] = int(target_box[0])
                    target_list[id][2] = int(target_box[1])
                    target_list[id][3] = int(target_box[2])
                    target_list[id][4] = int(target_box[3])

                # print(image_cur_path + " Target "+ str(id) + " x: " + str(target_box[0]) + " y: " + str(target_box[1]) + " w: " + str(target_box[2]) + " h: " + str(target_box[3]))  

        print(target_list)
        print(dirs[dir_num])
        for target in target_list:
            f.write(str(dirs[dir_num].split('.')[0][5:]))
            f.write(', ')
            for info in target[:-1]:
                f.write(str(info))
                f.write(', ')
            f.write(str(target[-1]))
            f.write('\n')
        draw_target(image, target_list)
        cv2.imwrite(args.folder + "_track/"+dir, image)    
    f.close()

def main():
    global classes, COLORS
    global target_x, target_y, target_w, target_h, target_id

    if (args.folder == "level1"):
        start = time.time()
        level1()
        print(args.folder + " time: " + str(time.time()-start))
    if (args.folder == "level2"):
        start = time.time()
        level2()
        print(args.folder + " time: " + str(time.time()-start))
    

main()
