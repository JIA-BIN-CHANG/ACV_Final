import os
import argparse
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--folder', required=True,
                help = 'path to the folder need to be detected')
args = ap.parse_args()

def id(name):
    num = name.split(',')[1]
    return int(num)

def get_iou(bbox_ai, bbox_gt):
    iou_x = max(bbox_ai[0], bbox_gt[0]) # x
    iou_y = max(bbox_ai[1], bbox_gt[1]) # y
    iou_w = min(bbox_ai[2]+bbox_ai[0], bbox_gt[2]+bbox_gt[0]) - iou_x # w
    iou_w = max(iou_w, 0)
    # print(f'{iou_w=}')
    iou_h = min(bbox_ai[3]+bbox_ai[1], bbox_gt[3]+bbox_gt[1]) - iou_y # h
    iou_h = max(iou_h, 0)
    # print(f'{iou_h=}')

    iou_area = iou_w * iou_h
    # print(f'{iou_area=}')
    all_area = bbox_ai[2]*bbox_ai[3] + bbox_gt[2]*bbox_gt[3] - iou_area
    # print(f'{all_area=}')

    return max(iou_area/all_area, 0)


def level12():
    gt_path = args.folder + ".txt"
    detect_path = args.folder + "_result.txt"

    gt_f = open(gt_path, 'r')
    gt_list = []
    for line in gt_f.readlines():
        gt_list.append(line)
    gt_f.close()

    detect_f = open(detect_path, 'r')
    detect_list = []
    for line in detect_f.readlines():
        detect_list.append(line)
    detect_f.close()

    detect_list.sort(key=id)

    # print(gt_list[:10])
    # print(detect_list[:10])

    frame_seq = []
    frame_label = []
    frame_iou = []
    for num in range(len(detect_list)):
        frame_seq.append(int(detect_list[num].split(',')[0]))
        frame_label.append(int(detect_list[num].split(',')[1]))
        frame_iou.append(get_iou(list(map(int, detect_list[num].rstrip('\n').split(',')[2:6])), list(map(int, gt_list[num].rstrip('\n').split(',')[2:6]))))

    print(frame_seq[:20])
    print(frame_iou[:20])

    line = plt.plot(frame_seq, frame_iou)
    plt.legend(line, frame_label)
    plt.title(args.folder + " Tracking IoU")
    plt.xlabel("Frame")
    plt.ylabel("IoU")
    plt.savefig(args.folder + " Result.png")

def level34():
    gt_path = args.folder + ".txt"
    detect_path = args.folder + "_result.txt"

    gt_f = open(gt_path, 'r')
    gt_list = []
    for line in gt_f.readlines():
        gt_list.append(line)
    gt_f.close()

    detect_f = open(detect_path, 'r')
    detect_list = []
    for line in detect_f.readlines():
        detect_list.append(line)
    detect_f.close()

    detect_list.sort(key=id)
    print(len(detect_list))

    print(gt_list[:10])
    print(detect_list[:10])

    frame_seq = []
    frame_label_1 = []
    frame_label_2 = []
    frame_iou_1 = []
    frame_iou_2 = []
    for num in range(int(0.5*len(detect_list))):
        frame_seq.append(int(detect_list[num].split(',')[0]))
        frame_label_1.append(int(detect_list[num].split(',')[1]))
        frame_iou_1.append(get_iou(list(map(int, detect_list[num].rstrip('\n').split(',')[2:6])), list(map(int, gt_list[num].rstrip('\n').split(',')[2:6]))))
    for num in range(int(0.5*len(detect_list)), len(detect_list)):
        # frame_seq.append(int(detect_list[num].split(',')[0]))
        frame_label_2.append(int(detect_list[num].split(',')[1]))
        frame_iou_2.append(get_iou(list(map(int, detect_list[num].rstrip('\n').split(',')[2:6])), list(map(int, gt_list[num].rstrip('\n').split(',')[2:6]))))

    # print(frame_seq[:20])
    # print(frame_iou_1[:20])

    plt.plot(frame_seq, frame_iou_1)
    plt.plot(frame_seq, frame_iou_2)
    plt.legend(['2', '6'])
    plt.title(args.folder + " Tracking IoU")
    plt.xlabel("Frame")
    plt.ylabel("IoU")
    plt.savefig(args.folder + " Result.png")

def level56():
    gt_path = args.folder + ".txt"
    detect_path = args.folder + "_result.txt"

    gt_f = open(gt_path, 'r')
    gt_list = []
    for line in gt_f.readlines():
        gt_list.append(line)
    gt_f.close()

    detect_f = open(detect_path, 'r')
    detect_list = []
    for line in detect_f.readlines():
        detect_list.append(line)
    detect_f.close()

    detect_list.sort(key=id)

    # print(gt_list[:10])
    # print(len(detect_list))

    frame_seq = []
    frame_label_1 = []
    frame_label_2 = []
    frame_label_3 = []
    frame_label_4 = []
    frame_iou_1 = []
    frame_iou_2 = []
    frame_iou_3 = []
    frame_iou_4 = []
    for num in range(int(0.25*len(detect_list))):
        frame_seq.append(int(detect_list[num].split(',')[0]))
        frame_label_1.append(int(detect_list[num].split(',')[1]))
        frame_iou_1.append(get_iou(list(map(int, detect_list[num].rstrip('\n').split(',')[2:6])), list(map(int, gt_list[num].rstrip('\n').split(',')[2:6]))))
    for num in range(int(0.25*len(detect_list)), int(0.5*len(detect_list))):
        # frame_seq.append(int(detect_list[num].split(',')[0]))
        frame_label_2.append(int(detect_list[num].split(',')[1]))
        frame_iou_2.append(get_iou(list(map(int, detect_list[num].rstrip('\n').split(',')[2:6])), list(map(int, gt_list[num].rstrip('\n').split(',')[2:6]))))
    for num in range(int(0.5*len(detect_list)), int(0.75*len(detect_list))):
        # frame_seq.append(int(detect_list[num].split(',')[0]))
        frame_label_3.append(int(detect_list[num].split(',')[1]))
        frame_iou_3.append(get_iou(list(map(int, detect_list[num].rstrip('\n').split(',')[2:6])), list(map(int, gt_list[num].rstrip('\n').split(',')[2:6]))))
    for num in range(int(0.75*len(detect_list)), int(len(detect_list))):
        # frame_seq.append(int(detect_list[num].split(',')[0]))
        frame_label_4.append(int(detect_list[num].split(',')[1]))
        frame_iou_4.append(get_iou(list(map(int, detect_list[num].rstrip('\n').split(',')[2:6])), list(map(int, gt_list[num].rstrip('\n').split(',')[2:6]))))

    # print(frame_seq[:20])
    # print(frame_iou[:20])

    plt.plot(frame_seq, frame_iou_1)
    plt.plot(frame_seq, frame_iou_2)
    plt.plot(frame_seq, frame_iou_3)
    plt.plot(frame_seq, frame_iou_4)
    plt.legend(['11', '12', '13', '29'])
    plt.title(args.folder + " Tracking IoU")
    plt.xlabel("Frame")
    plt.ylabel("IoU")
    plt.savefig(args.folder + " Result.png")

def main():
    if (args.folder == "level1" or args.folder == "level2"):
        level12()
    if (args.folder == "level3" or args.folder == "level4"):
        level34()
    if (args.folder == "level5" or args.folder == "level6"):
        level56()

main()