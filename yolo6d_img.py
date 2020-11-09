""" initial code for still images adapted from valid.py """
""" TO Test batch of still images
$ create conda environment -- similar to yolact environment
$ conda activate yolo6d
$ python3 fileName.py
"""
import os
import cv2
import time
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy.io
from skimage.transform import resize

import PIL.Image as pImage

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

from darknet import Darknet
import dataset
from utils import *
from MeshPly import MeshPly

# Create new directory
def makedirs(path):
    if not os.path.exists( path ):
        os.makedirs( path )

def valid(datacfg, modelcfg, weightfile):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    # Parse configuration files
    options      = read_data_cfg(datacfg)
    dataDir      = options['dataDir']
    meshname     = options['mesh']
    backupdir    = options['backup']
    name         = options['name']
    filetype     = options['rgbfileType']
    fx           = float(options['fx'])
    fy           = float(options['fy'])
    u0           = float(options['u0'])
    v0           = float(options['v0'])

    if not os.path.exists(backupdir):
        makedirs(backupdir)

    # Parameters
    prefix       = 'results'
    seed         = int(time.time())
    gpus         = options['gpus'] #'0' # Specify which gpus to use
    test_width   = 640 #672
    test_height  = 480 #672
    torch.manual_seed(seed)

    use_cuda = True
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)
    use_cuda        = True
    visualize       = True
    num_classes     = 1
    conf_thresh     = 0.3
    # nms_thresh      = 0.4
    # match_thresh    = 0.5

    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]

    # Read object model information, get 3D bounding box corners
    mesh          = MeshPly(meshname)
    vertices      = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D     = get_3D_corners(vertices)
    diam          = float(options['diam'])

    # Read intrinsic camera parameters
    internal_calibration = get_camera_intrinsic(u0, v0, fx, fy)

    # Specicy model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(modelcfg)
    model.load_weights(weightfile)
    model.cuda()
    model.eval()

    # Get the parser for the test dataset
    valid_dataset = dataset.listDataset(dataDir, filetype, shape=(test_width, test_height),
                        shuffle=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),]))
    valid_batchsize = 1

    # Specify the number of workers for multiple processing, get the dataloader for the test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=True, **kwargs)

    logging("   Testing {}...".format(name))
    logging("   Number of test samples: %d" % len(test_loader.dataset))

    # Iterate through test batches (Batch size for test data is 1)
    count = 0
    z = np.zeros((3, 1))
    for batch_idx, (data, target) in enumerate(test_loader):

        # Images
        img = data[0, :, :, :]
        img = img.numpy().squeeze()
        img = np.transpose(img, (1, 2, 0))
        # cv2.imshow('yolo6d', img)
        # cv2.waitKey(1000)

        t1 = time.time()
        # Pass data to GPU
        if use_cuda:
            data = data.cuda()
            target = target.cuda()

        # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
        data = Variable(data, volatile=True)
        t2 = time.time()

        # Forward pass
        output = model(data).data
        t3 = time.time()

        # Using confidence threshold, eliminate low-confidence predictions
        all_boxes = get_region_boxes2(output, conf_thresh, num_classes)
        # boxes = do_detect(model, frame_cp, 0.1, 0.4)
        t4 = time.time()

        # Iterate through all images in the batch
        for i in range(output.size(0)):

            allObj = []

            # If the prediction has the highest confidence, choose it as our prediction for single object pose estimation
            best_conf_est = -1

            # For each image, get all the predictions
            boxes   = all_boxes[i]
            print(len(boxes)-1, 'objects found')
            for j in range(len(boxes)-1):

                # ignore 1st box (NOTE: not sure why its incorrect)
                box_pr        = boxes[j+1]

                # Denormalize the corner predictions
                corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
                corners2D_pr[:, 0] = corners2D_pr[:, 0] * 640
                corners2D_pr[:, 1] = corners2D_pr[:, 1] * 480

                # Compute [R|t] by pnp
                R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(internal_calibration, dtype='float32'))

                # Compute pixel error
                Rt_pr        = np.concatenate((R_pr, t_pr), axis=1)
                proj_corners_pr = np.transpose(compute_projection(corners3D, Rt_pr, internal_calibration))

                # print('proj_corners_pr\n', proj_corners_pr)
                allObj.append(proj_corners_pr)

            t5 = time.time()

        # Visualize
        if visualize:
            plt.xlim((0, 640))
            plt.ylim((0, 480))
            plt.imshow(img)

            # Projections
            for obj in allObj:
                for edge in edges_corners:
                    plt.plot(obj[edge, 0],
                            obj[edge, 1],
                            color='r', linewidth=1.0)

            plt.gca().invert_yaxis()
            plt.show()


if __name__ == '__main__':

    datacfg = 'objects_cfg/txonigiri-test.data'
    modelcfg = 'models_cfg/tekin/yolo-pose.cfg'
    weightfile = 'backup/txonigiri/modelv4.3.weights' #v3.2(95.24%) < v4.1(95.87%) < v4.2(97.14%) == v4.3
    valid(datacfg, modelcfg, weightfile)