import cv2
import os
import argparse

def getint(name):
    num = name.split('.')[0]
    num = num[5:]
    return int(num)

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--folder', required=True,
                help = 'path to the folder need to be detected')
args = ap.parse_args()

fourcc = cv2.VideoWriter_fourcc(*'XVID')

output_name = args.folder + '_result.avi'
print(output_name)
out = cv2.VideoWriter(output_name, fourcc, 30.0, (1920, 1080))


track_dir = args.folder + '_track'
os.chdir(os.path.join(track_dir))
path = os.path.join(os.getcwd())

data_list = os.listdir(path)
data_list.sort(key=getint)

for frame in data_list:
    frame = path + "/" + frame
    print(frame)
    image = cv2.imread(frame)
    # cv2.imshow("myimage",image)
    out.write(image)