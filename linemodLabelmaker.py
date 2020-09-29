""" convert pose from ndds JSON to Linemod YML format """

import os
import sys
import time
import json
import yaml
import shutil
import getpass
import cv2
import numpy as np
import random as rand
from scipy.spatial.transform import Rotation as R

username = getpass.getuser()
osName = os.name
if osName == 'posix':
    os.system('clear')
else:
    os.system('cls')


def draw_box(x, y, w, h, img):

    x = int(x)
    y = int(y)
    w = x + w
    h = y + h
    img = cv2.rectangle(img, (x, y), (w, h), (255*rand.uniform(0,1), 55*rand.uniform(0,1), 125*rand.uniform(0,1)), 2)
    cv2.imshow('ndds bbox', img)
    cv2.waitKey(1000)


def ndds2linemod(img, jsonfile):

# 21 numbers correspond to the following:
# 1st number: class label,
# 2nd number: x0 (x-coordinate of the centroid), 3rd number: y0 (y-coordinate of the centroid),
# 4th number: x1 (x-coordinate of the first corner), 5th number: y1 (y-coordinate of the first corner
# 18th number: x8 (x-coordinate of the eighth corner), 19th number: y8 (y-coordinate of the eighth corner),
# 20th number: x range, 21st number: y range of bounding box

    points2D = []

    with open(str(jsonfile)) as attributes:
        data = json.load(attributes)

    # extract projected points, bounding box
    for obj in range(len(data["objects"])):

        bbox = data['objects'][obj]['bounding_box']

        # NOTE: remember ndds x,y are reversed
        bbox_x = round(bbox['top_left'][1])
        bbox_y = round(bbox['top_left'][0])
        bbox_w = round(bbox['bottom_right'][1]) - bbox_x
        bbox_h = round(bbox['bottom_right'][0]) - bbox_y

        xywh = [bbox_x, bbox_y, bbox_w, bbox_h]
        # print(xywh)

        img = cv2.imread(img, 0)
        height, width = img.shape

        # visualize ndds bbox
        # draw_box(bbox_x, bbox_y, bbox_w, bbox_h, img)

        # 1st class label
        points2D.append(0)

        # NOTE: remember ndds x,y are reversed
        # 2nd & 3rd centroid
        points2D.append(data['objects'][obj]['projected_cuboid_centroid'][1]/width)  #x
        points2D.append(data['objects'][obj]['projected_cuboid_centroid'][0]/height) #y

        # 4th to 19th
        for i in range(len(data['objects'][obj]['projected_cuboid'])):
            points2D.append(data['objects'][obj]['projected_cuboid'][i][1]/width)  #x
            points2D.append(data['objects'][obj]['projected_cuboid'][i][0]/height) #y

        # 20th and 21st
        points2D.append(bbox_w/width)
        points2D.append(bbox_h/height)

    return points2D

if __name__ == "__main__":

    startTime = time.time()

    path = '/media/ash/SSD/Odaiba/dataset/densefusion/txonigiri/data/01/'

    # whichFile train or test
    whichFile = ['train.txt', 'test.txt']

    for f in files:
        openfile = open(path + whichFile[f], 'r')
        lines = openfile.readlines()

        count = 0
        for line in lines:
            print( "Line {}: {}".format( count+1, int( line.strip() ) ) )

            imagefile = path + 'rgb/'  + line.strip() + '.png'
            jsonfile  = path + 'json/' + line.strip() + '.json'
            labelfile = path + 'label/' + line.strip() + '.txt'

            open(labelfile, 'w').close() # erase old file
            writefile = open(labelfile, 'w')

            points2D = ndds2linemod(imagefile, jsonfile)
            for value in range(len(points2D)):
                # print(str(points2D[value]))
                writefile.write(str(points2D[value]))
                if value < len(points2D)-1:
                    writefile.write(' ')
            writefile.close()

            count +=1
            # if count >1:
                # break