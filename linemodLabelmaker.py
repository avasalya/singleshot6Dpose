""" convert pose from ndds JSON to Linemod YML format """

import os
import sys
import json
import yaml
import shutil
import getpass
import cv2
import numpy as np
import random as rand
import PIL.Image as pImage
from PIL import ImageDraw
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
    img = cv2.rectangle(img, (x, y), (w, h), (255,0,0), 2)
    # cv2.imshow('ndds bbox', img)

def draw_cube(projPts, img, color, lineWidth):

    p0 = (int((projPts[0][0])), int((projPts[0][1])))
    p1 = (int((projPts[1][0])), int((projPts[1][1])))
    p2 = (int((projPts[2][0])), int((projPts[2][1])))
    p3 = (int((projPts[3][0])), int((projPts[3][1])))
    p4 = (int((projPts[4][0])), int((projPts[4][1])))
    p5 = (int((projPts[5][0])), int((projPts[5][1])))
    p6 = (int((projPts[6][0])), int((projPts[6][1])))
    p7 = (int((projPts[7][0])), int((projPts[7][1])))

    cv2.line(img, p0, p1, color, lineWidth)
    cv2.line(img, p0, p3, color, lineWidth)
    cv2.line(img, p0, p4, color, lineWidth)
    cv2.line(img, p1, p2, color, lineWidth)
    cv2.line(img, p1, p5, color, lineWidth)
    cv2.line(img, p2, p3, color, lineWidth)
    cv2.line(img, p2, p6, color, lineWidth)
    cv2.line(img, p3, p7, color, lineWidth)
    cv2.line(img, p4, p5, color, lineWidth)
    cv2.line(img, p4, p7, color, lineWidth)
    cv2.line(img, p5, p6, color, lineWidth)
    cv2.line(img, p6, p7, color, lineWidth)

def ndds2linemod(img, jsonfile):
    """
        # 21 numbers correspond to the following:
        # 1st:  class label/index
        # 2nd:  x0 (x-coordinate of the centroid)      3rd:  y0 (y-coordinate of the centroid)
        # 4th:  x1 (x-coordinate of the first corner)  5th:  y1 (y-coordinate of the first corner)
        # 18th: x8 (x-coordinate of the eighth corner) 19th: y8 (y-coordinate of the eighth corner)
        # 20th: x range                                21st: y range of bounding box
    """
    points2D = []
    points2DCheck = []

    xIndex = 0
    yIndex = 1

    with open(str(jsonfile)) as attributes:
        data = json.load(attributes)

    # extract projected points, bounding box
    for obj in range(len(data["objects"])):

        # NOTE: remember ndds x,y are reversed NOTE: ONLY fo BoundingBox
        bbox = data['objects'][obj]['bounding_box']
        bbox_x = round(bbox['top_left'][yIndex])
        bbox_y = round(bbox['top_left'][xIndex])
        bbox_w = round(bbox['bottom_right'][yIndex]) - bbox_x
        bbox_h = round(bbox['bottom_right'][xIndex]) - bbox_y
        # print([bbox_x, bbox_y, bbox_w, bbox_h])

        # display rgb image
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        # print('height {}, width {}'.format(height, width))

        # draw ndds bbox
        draw_box(bbox_x, bbox_y, bbox_w, bbox_h, img)

        # draw 2D projected cube
        projPts = data['objects'][obj]['projected_cuboid']
        print("2D projected points from NDDS\n", data['objects'][obj]['projected_cuboid'],'\n')
        draw_cube(projPts, img, (0, 255, 0), 2)

        # 1st 'class label/index'
        points2D.append(0)

        # NOTE: normalize points befor making a list to save as labels
        # 2nd & 3rd centroid
        points2D.append(data['objects'][obj]['projected_cuboid_centroid'][xIndex]/width)  #x
        points2D.append(data['objects'][obj]['projected_cuboid_centroid'][yIndex]/height) #y

        # 4th to 19th
        for i in range(len(data['objects'][obj]['projected_cuboid'])):
            points2D.append(data['objects'][obj]['projected_cuboid'][i][xIndex]/width)  #x
            points2D.append(data['objects'][obj]['projected_cuboid'][i][yIndex]/height) #y

            newList = []
            newList.append(data['objects'][obj]['projected_cuboid'][i][xIndex]) #x
            newList.append(data['objects'][obj]['projected_cuboid'][i][yIndex]) #y
            points2DCheck.append(newList)

        # 20th and 21stx
        points2D.append(bbox_w/width)
        points2D.append(bbox_h/height)

        # cross-check projected cuboid points
        print("cross-check 2d cuboids\n", points2DCheck, '\n')
        draw_cube(points2DCheck, img, (0, 0, 255), 1)

        # visualize output
        cv2.imshow('ndds bbox', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(100)

    return points2D

if __name__ == "__main__":

    path = '/media/ash/SSD/Odaiba/dataset/linemod-onigiri/txonigiri/data/01/'

    # whichFile train or test
    # whichFile = ['test.txt']
    whichFile = ['train.txt', 'test.txt']

    for labelFile in whichFile:
        openfile = open(path + labelFile, 'r')
        lines = openfile.readlines()

        count = 0
        for line in lines:
            print( "Line {}: {}".format( count+1, int( line.strip() ) ) )

            imagefile = path + 'JPEGImages/'  + line.strip() + '.png'
            jsonfile  = path + 'json/' + line.strip() + '.json'
            labelfile = path + 'labels/' + line.strip() + '.txt'
            # labelfile = path + 'dummylabels/' + line.strip() + '.txt'

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
            if count >1000000:
                break