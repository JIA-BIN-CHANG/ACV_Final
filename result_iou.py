import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--folder', required=True,
                help = 'path to the folder need to be detected')
args = ap.parse_args()

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

print(gt_list[:2])
print(detect_list[0].split(',')[1])