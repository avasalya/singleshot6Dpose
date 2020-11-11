""" TO Test still images one at time
$ create conda environment -- similar to yolact environment
$ conda activate yolo6d
$ python3 fileName.py
"""
import os
import sys
import cv2
import time
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy.io
from skimage.transform import resize

import PIL.Image as Image

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

from darknet import Darknet
import dataset
from utils import *
from MeshPly import MeshPly


def valid(datacfg, modelcfg, weightfile):

    # Parameters
    options    = read_data_cfg(datacfg)
    dataDir    = options['dataDir']
    meshname   = options['mesh']
    name       = options['name']
    filetype   = options['rgbfileType']
    fx         = float(options['fx'])
    fy         = float(options['fy'])
    u0         = float(options['u0'])
    v0         = float(options['v0'])
    seed       = int(time.time())
    gpus       = options['gpus']
    img_width  = 640
    img_height = 480
    torch.manual_seed(seed)

    use_cuda = True
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)

    visualize    = True
    num_classes  = 1
    conf_thresh  = 0.5
    # nms_thresh   = 0.4
    # match_thresh = 0.5

    # Read object model information, get 3D bounding box corners
    mesh      = MeshPly(meshname)
    vertices  = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D = get_3D_corners(vertices)

    # Read intrinsic camera parameters
    internal_calibration = get_camera_intrinsic(u0, v0, fx, fy)

    # Specify model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(modelcfg)
    model.load_weights(weightfile)
    model.cuda()
    model.eval()

    # apply transformation on the input images
    transform = transforms.Compose([transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

    # read still images as per the test set
    with open(os.path.join(dataDir, 'test.txt'), 'r') as file:
        lines = file.readlines()
    imgindex = lines[2].rstrip()
    imgpath = os.path.join(dataDir, 'rgb', str(imgindex) + filetype)

    # read image for visualization
    img = cv2.imread(imgpath)
    # cv2.imshow('yolo6d', img), # cv2.waitKey(1)

    # read images usin PIL
    img_ = Image.open(imgpath).convert('RGB')
    img_ = img_.resize((img_width, img_height))
    t1 = time.time()

    # transform into Tensor
    img_ = transform(img_)
    data = Variable(img_).cuda().unsqueeze(0)
    t2 = time.time()

    # Forward pass
    output = model(data).data
    t3 = time.time()

    # Using confidence threshold, eliminate low-confidence predictions
    all_boxes = get_region_boxes2(output, conf_thresh, num_classes)
    # all_boxes = do_detect(model, img, 0.1, 0.4)
    t4 = time.time()

    # For each image, get all the predictions
    allBoxes = []
    boxes = all_boxes[0]
    print(len(boxes)-1, 'onigiri(s) found')
    for j in range(len(boxes)-1):

        # ignore 1st box (NOTE: not sure why its incorrect)
        box_pr = boxes[j+1]

        # Denormalize the corner predictions
        corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
        corners2D_pr[:, 0] = corners2D_pr[:, 0] * img_width
        corners2D_pr[:, 1] = corners2D_pr[:, 1] * img_height

        # Compute [R|t] by PnP
        R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(internal_calibration, dtype='float32'))
        Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
        proj_corners_pr = np.transpose(compute_projection(corners3D, Rt_pr, internal_calibration))

        allBoxes.append(proj_corners_pr)

    t5 = time.time()

    # Visualize
    if visualize:
        # Projections
        for corner in allBoxes:
            color     = (0,0,255)
            linewidth = 2
            img = cv2.line(img, tuple(corner[0]), tuple(corner[1]), color, linewidth)
            img = cv2.line(img, tuple(corner[0]), tuple(corner[2]), color, linewidth)
            img = cv2.line(img, tuple(corner[0]), tuple(corner[4]), color, linewidth)
            img = cv2.line(img, tuple(corner[1]), tuple(corner[3]), color, linewidth)
            img = cv2.line(img, tuple(corner[1]), tuple(corner[5]), color, linewidth)
            img = cv2.line(img, tuple(corner[2]), tuple(corner[3]), color, linewidth)
            img = cv2.line(img, tuple(corner[2]), tuple(corner[6]), color, linewidth)
            img = cv2.line(img, tuple(corner[3]), tuple(corner[7]), color, linewidth)
            img = cv2.line(img, tuple(corner[4]), tuple(corner[5]), color, linewidth)
            img = cv2.line(img, tuple(corner[4]), tuple(corner[6]), color, linewidth)
            img = cv2.line(img, tuple(corner[5]), tuple(corner[7]), color, linewidth)
            img = cv2.line(img, tuple(corner[6]), tuple(corner[7]), color, linewidth)
        cv2.imshow('yolo6d pose', img)
        key = cv2.waitKey(10000) & 0xFF
        if key == 27:
            print('stopping, keyboard interrupt')
            sys.exit()

if __name__ == '__main__':

    datacfg = 'objects_cfg/txonigiri-test.data'
    modelcfg = 'models_cfg/tekin/yolo-pose.cfg'
    weightfile = 'backup/txonigiri/modelv4.3.weights' #v3.2(95.24%) < v4.1(95.87%) < v4.2(97.14%) == v4.3
    valid(datacfg, modelcfg, weightfile)