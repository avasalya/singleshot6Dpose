from utils import *
import cv2
import numpy as np
from darknet import Darknet
from glob import glob
import torch

def detect(input_path, weightsfile, cfgfile, show_detection = True, save_video = False, save_images = False):
    # Parameters
    seed         = int(time.time())
    gpus         = '0'     # Specify which gpus to use
    torch.manual_seed(seed)
    use_cuda = True
    if use_cuda:
        torch.device(0)
        torch.cuda.manual_seed(seed)

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    model.cuda()

    images_path = glob(input_path + '/*')
    
    # Get size of input image
    height, width, _ = cv2.imread(images_path[0]).shape
    if save_video:
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (width, height))

    for file_name in images_path:
        frame = cv2.imread(file_name)
        frame_cp = frame.copy()
        frame_cp = cv2.resize(frame_cp, (416, 416))
        boxes = do_detect(model, frame_cp, 0.1, 0.4)

        best_confidence_pr = -1
        box_pr = None

        for b in range(len(boxes)):
            if boxes[b][18] > best_confidence_pr:
                box_pr = boxes[b]
                best_confidence_pr = boxes[b][18]


        # get corners
        corners_2d_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
        corners_2d_pr[:, 0] = corners_2d_pr[:, 0] * frame.shape[1]
        corners_2d_pr[:, 1] = corners_2d_pr[:, 1] * frame.shape[0]

        frame = draw_bb(frame, corners_2d_pr)
        if show_detection:
            cv2.imshow('Frame', frame)
        if save_video:
            out.write(frame)
        cv2.waitKey(1)

    if save_video:
        out.release()

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 4:
        input_path = sys.argv[1]
        cfgfile = sys.argv[2]
        weightsfile = sys.argv[3]
        detect(input_path, weightsfile, cfgfile)
    else:
        print('Usage:')
        print(' python detect.py input_folder weightfile cfgfile')
        print("Example:")
        print("python detect.py LINEMOD/ape/JPEGImages cfg/yolo-pose.cfg backup/ape/model_backup.weights")