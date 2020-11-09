""" YOLO6D ROS Wrapper """
""" TO Test
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



class Yolo6D:

    def __init__(self, datacfg, modelcfg, weightfile):
        # Parameters
        options          = read_data_cfg(datacfg)
        self.dataDir     = options['dataDir']
        meshname         = options['mesh']
        filetype         = options['rgbfileType'] #no need later
        fx               = float(options['fx'])
        fy               = float(options['fy'])
        u0               = float(options['u0'])
        v0               = float(options['v0'])
        gpus             = options['gpus']
        seed             = int(time.time())
        use_cuda         = True
        self.img_width   = 640
        self.img_height  = 480
        self.classes     = 1
        self.conf_thresh = 0.5
        # nms_thresh       = 0.4
        # match_thresh     = 0.5

        torch.manual_seed(seed)
        if use_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            torch.cuda.manual_seed(seed)

        # Read intrinsic camera parameters
        self.internal_calibration = get_camera_intrinsic(u0, v0, fx, fy)

        # Read object model information, get 3D bounding box corners
        mesh      = MeshPly(meshname)
        vertices  = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
        self.corners3D = get_3D_corners(vertices)

        # Specify model, load pretrained weights, pass to GPU and set the module in evaluation mode
        self.model = Darknet(modelcfg)
        self.model.load_weights(weightfile)
        self.model.cuda()
        self.model.eval()

        # apply transformation on the input images
        self.transform = transforms.Compose([transforms.ToTensor()])

        # read still images as per the test set
        with open(os.path.join(self.dataDir, 'test.txt'), 'r') as file:
            lines = file.readlines()
        imgindex = lines[2].rstrip()
        imgpath = os.path.join(self.dataDir, 'rgb', str(imgindex) + filetype)

        # read image for visualization
        self.img = cv2.imread(imgpath)
        # cv2.imshow('yolo6d', self.img), # cv2.waitKey(1)

        # read images usin PIL
        self.img_ = Image.open(imgpath).convert('RGB')
        self.img_ = self.img_.resize((self.img_width, self.img_height))
        t1 = time.time()


    def pose_estimator(self):
        # transform into Tensor
        self.img_ = self.transform(self.img_)
        data = Variable(self.img_).cuda().unsqueeze(0)
        t2 = time.time()

        # Forward pass
        output = self.model(data).data
        t3 = time.time()

        # Using confidence threshold, eliminate low-confidence predictions
        self.all_boxes = get_region_boxes2(output, self.conf_thresh, self.classes)
        print('size all_boxes', len(self.all_boxes))
        # all_boxes = do_detect(self.model, self.img, 0.1, 0.4)
        t4 = time.time()

        # For each image, get all the predictions
        boxesList = []
        boxes = self.all_boxes[0]
        print(len(boxes)-1, 'onigiri(s) found')
        for j in range(len(boxes)-1):

            # ignore 1st box (NOTE: not sure why its incorrect)
            box_pr = boxes[j+1]

            # Denormalize the corner predictions
            corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
            corners2D_pr[:, 0] = corners2D_pr[:, 0] * self.img_width
            corners2D_pr[:, 1] = corners2D_pr[:, 1] * self.img_height

            # Compute [R|t] by PnP
            R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), self.corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(self.internal_calibration, dtype='float32'))
            Rt_pr = np.concatenate((R_pr, t_pr), axis=1)

            # Compute projections
            proj_corners_pr = np.transpose(compute_projection(self.corners3D, Rt_pr, self.internal_calibration))
            boxesList.append(proj_corners_pr)

        t5 = time.time()

        # Visualize Projections
        self.visualize(self.img, boxesList, drawCuboid=True)


    def visualize(self, img, boxesList, drawCuboid=True):
        if drawCuboid:
            for corner in boxesList:
                img = cv2.line(img, tuple(corner[0]), tuple(corner[1]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[0]), tuple(corner[2]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[0]), tuple(corner[4]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[1]), tuple(corner[3]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[1]), tuple(corner[5]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[2]), tuple(corner[3]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[2]), tuple(corner[6]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[3]), tuple(corner[7]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[4]), tuple(corner[5]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[4]), tuple(corner[6]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[5]), tuple(corner[7]), (0,0,255), 1)
                img = cv2.line(img, tuple(corner[6]), tuple(corner[7]), (0,0,255), 1)
        cv2.imshow('yolo6d pose', img)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            print('stopping, keyboard interrupt')
            sys.exit()



if __name__ == '__main__':

    # rospy.init_node('onigiriPose', anonymous=False)
    # rospy.loginfo('starting onigiriPose node....')

    datacfg = 'objects_cfg/txonigiri-test.data'
    modelcfg = 'models_cfg/tekin/yolo-pose.cfg'
    weightfile = 'backup/txonigiri/modelv4.3.weights' #v3.2(95.24%) < v4.1(95.87%) < v4.2(97.14%) == v4.3

    """ run Yolo6D """
    y6d = Yolo6D(datacfg, modelcfg, weightfile)
    y6d.pose_estimator()

