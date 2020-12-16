
"""(yolo6d) python3 filename.py """
"""(dfros3) python filename.py """

import os
import cv2
import time
import warnings
import argparse

import scipy.io
import scipy.misc
import pandas as pd
import matplotlib.pyplot as plt

from MeshPly import MeshPly

import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

import dataset
from utils import *
from darknet import Darknet


# Create new directory
def makedirs_(path):
    if not os.path.exists( path ):
        os.makedirs( path )

def truths_length(truths, max_num_gt=50):
    for i in range(max_num_gt):
        if truths[i][1] == 0:
            return i

def draw(img, axesPoint, cuboid, color):
        img = cv2.line(img, tuple(axesPoint[3].ravel()),
                            tuple(axesPoint[0].ravel()), (255,0,0), 2)
        img = cv2.line(img, tuple(axesPoint[3].ravel()),
                            tuple(axesPoint[1].ravel()), (0,255,0), 2)
        img = cv2.line(img, tuple(axesPoint[3].ravel()),
                            tuple(axesPoint[2].ravel()), (0,0,255), 2)
        cv2.circle(img, tuple(axesPoint[3].ravel()), 5, (0, 255, 255), -1)

        cv2.line(img, cuboid[0], cuboid[1], color, 2)
        cv2.line(img, cuboid[0], cuboid[2], color, 2)
        cv2.line(img, cuboid[0], cuboid[4], color, 2)
        cv2.line(img, cuboid[1], cuboid[3], color, 2)
        cv2.line(img, cuboid[1], cuboid[5], color, 2)
        cv2.line(img, cuboid[2], cuboid[3], color, 2)
        cv2.line(img, cuboid[2], cuboid[6], color, 2)
        cv2.line(img, cuboid[3], cuboid[7], color, 2)
        cv2.line(img, cuboid[4], cuboid[5], color, 2)
        cv2.line(img, cuboid[4], cuboid[6], color, 2)
        cv2.line(img, cuboid[5], cuboid[7], color, 2)
        cv2.line(img, cuboid[6], cuboid[7], color, 2)


def valid(datacfg, modelcfg):

    # Parse configuration files
    data_options = read_data_cfg(datacfg)
    dataDir      = data_options['dataDir']
    weightfile   = data_options['weightfile']
    meshname     = data_options['mesh']
    backupdir    = data_options['backup']
    name         = data_options['name']
    gpus         = data_options['gpus']
    fx           = float(data_options['fx'])
    fy           = float(data_options['fy'])
    cx           = float(data_options['cx'])
    cy           = float(data_options['cy'])
    im_width     = int(data_options['width'])
    im_height    = int(data_options['height'])

    if not os.path.exists(backupdir):
        makedirs_(backupdir)

    # Parameters
    seed = int(time.time())
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)
    save            = False
    testtime        = True
    visualize       = True
    num_classes     = 1
    testing_samples = 0.0
    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]

    if save:
        makedirs_(backupdir + '/test')
        makedirs_(backupdir + '/test/gt')
        makedirs_(backupdir + '/test/pr')
        print("testing directories created inside backup folder")

    # To save
    testing_error_trans = 0.0
    testing_error_angle = 0.0
    testing_error_pixel = 0.0
    errs_2d             = []
    errs_3d             = []
    errs_trans          = []
    errs_angle          = []
    errs_corner2D       = []
    preds_trans         = []
    preds_rot           = []
    preds_corners2D     = []
    gts_trans           = []
    gts_rot             = []
    gts_corners2D       = []

    # Read object model information, get 3D bounding box corners
    mesh      = MeshPly(meshname)
    vertices  = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D = get_3D_corners(vertices)
    try:
        diam  = float(data_options['diam'])
    except:
        diam  = calc_pts_diameter(np.array(mesh.vertices)) #this takes too much time

    # Read intrinsic camera parameters
    intrinsic_calibration = get_camera_intrinsic(cx, cy, fx, fy)

    # Get validation file names
    valid_images = os.path.join(dataDir + 'test.txt')
    with open(valid_images) as fp:
        print("getting validation files")
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]

    # Specify model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(modelcfg, distiling=distiling)
    # model.print_network()
    model.load_weights(weightfile)
    model.cuda()
    model.eval()
    test_width    = model.test_width
    test_height   = model.test_height
    num_keypoints = model.num_keypoints
    num_labels    = num_keypoints * 2 + 3

    # Get the parser for the test dataset
    valid_dataset = dataset.listDataset(dataDir,
                                        shape=(test_width, test_height),
                                        shuffle=False,
                                        transform=transforms.Compose([transforms.ToTensor(),]))

    # Specify the number of workers for multiple processing, get the dataloader for the test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, **kwargs)

    logging("   Testing {}...".format(name))
    logging("   Number of test samples: %d" % len(test_loader.dataset))
    # Iterate through test batches (Batch size for test data is 1)
    count = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):

            # Images tensor
            img = data[0, :, :, :]

            # convert image tensor to numpy ndarray
            img = img.numpy().squeeze()

            # transpose image
            img = np.transpose(img, (1, 2, 0))

            t1 = time.time()
            # Pass data to GPU
            data = data.cuda()
            target = target.cuda()
            # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
            data = Variable(data)
            # data = Variable(data, volatile=True)
            t2 = time.time()
            # Forward pass
            output = model(data).data
            t3 = time.time()
            # Using confidence threshold, eliminate low-confidence predictions
            all_boxes = get_region_boxes(output, num_classes, num_keypoints)
            t4 = time.time()
            # Evaluation
            # Iterate through all batch elements
            for box_pr, target in zip([all_boxes], [target[0]]):
                # print('box pred', box_pr)
                # print('target', target)

                # For each image, get all the targets (for multiple object pose estimation, there might be more than 1 target per image)
                truths = target.view(-1, num_keypoints*2+3)
                # Get how many objects are present in the scene
                num_gts    = truths_length(truths)
                # Iterate through each ground-truth object
                for k in range(num_gts):
                    box_gt = list()
                    for j in range(1, 2*num_keypoints+1):
                        box_gt.append(truths[k][j])
                    box_gt.extend([1.0, 1.0])
                    box_gt.append(truths[k][0])

                    # Denormalize the corner predictions
                    corners2D_gt = np.array(np.reshape(box_gt[:18], [9, 2]), dtype='float32')
                    corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
                    corners2D_gt[:, 0] = corners2D_gt[:, 0] * im_width
                    corners2D_gt[:, 1] = corners2D_gt[:, 1] * im_height
                    corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width
                    corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height
                    preds_corners2D.append(corners2D_pr)
                    gts_corners2D.append(corners2D_gt)

                    # Compute corner prediction error
                    corner_norm = np.linalg.norm(corners2D_gt - corners2D_pr, axis=1)
                    corner_dist = np.mean(corner_norm)
                    errs_corner2D.append(corner_dist)

                    # Compute [R|t] by pnp
                    R_gt, t_gt = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_gt, np.array(intrinsic_calibration, dtype='float32'))
                    R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(intrinsic_calibration, dtype='float32'))

                    # Compute translation error
                    trans_dist   = np.sqrt(np.sum(np.square(t_gt - t_pr)))
                    errs_trans.append(trans_dist)

                    # Compute angle error
                    angle_dist   = calcAngularDistance(R_gt, R_pr)
                    errs_angle.append(angle_dist)

                    # Compute pixel error
                    Rt_gt        = np.concatenate((R_gt, t_gt), axis=1)
                    Rt_pr        = np.concatenate((R_pr, t_pr), axis=1)
                    proj_2d_gt   = compute_projection(vertices, Rt_gt, intrinsic_calibration)
                    proj_2d_pred = compute_projection(vertices, Rt_pr, intrinsic_calibration)
                    proj_corners_gt = np.transpose(compute_projection(corners3D, Rt_gt, intrinsic_calibration))
                    proj_corners_pr = np.transpose(compute_projection(corners3D, Rt_pr, intrinsic_calibration))
                    norm         = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
                    pixel_dist   = np.mean(norm)
                    errs_2d.append(pixel_dist)

                    # Visualize
                    if visualize:
                        img = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # draw axes & cuboid Projections
                        points = np.float32([[.1, 0, 0], [0, .1, 0], [0, 0, .1], [0, 0, 0]]).reshape(-1, 3)
                        axesPoints_gt = cv2.projectPoints(
                                                    points,
                                                    cv2.Rodrigues(R_gt)[0],
                                                    t_gt,
                                                    intrinsic_calibration,
                                                    None)[0]
                        cuboid_gt  = [tuple(map(int, point)) for point in proj_corners_gt]
                        draw(img, axesPoints_gt, cuboid_gt, (0,255,0))

                        axesPoints_pr = cv2.projectPoints(
                                                    points,
                                                    cv2.Rodrigues(R_pr)[0],
                                                    t_pr,
                                                    intrinsic_calibration,
                                                    None)[0]
                        cuboid_pr  = [tuple(map(int, point)) for point in proj_corners_pr]
                        draw(img, axesPoints_pr, cuboid_pr, (0,0,255))

                        cv2.imshow('validate image', img)
                        key = cv2.waitKey(1000) & 0xFF
                        if key == 27:
                            print('stopping, keyboard interrupt')
                            os._exit(0)



                    # Compute 3D distances
                    transform_3d_gt   = compute_transformation(vertices, Rt_gt)
                    transform_3d_pred = compute_transformation(vertices, Rt_pr)
                    norm3d            = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
                    vertex_dist       = np.mean(norm3d)
                    errs_3d.append(vertex_dist)

                    # Sum errors
                    testing_error_trans  += trans_dist
                    testing_error_angle  += angle_dist
                    testing_error_pixel  += pixel_dist
                    testing_samples      += 1
                    count = count + 1

                    if save:
                        preds_trans.append(t_pr)
                        gts_trans.append(t_gt)
                        preds_rot.append(R_pr)
                        gts_rot.append(R_gt)

                        # np.savetxt(backupdir + '/test/gt/R_' + valid_files[count][-8:-3] + 'txt', np.array(R_gt, dtype='float32'))
                        # np.savetxt(backupdir + '/test/gt/t_' + valid_files[count][-8:-3] + 'txt', np.array(t_gt, dtype='float32'))
                        # np.savetxt(backupdir + '/test/pr/R_' + valid_files[count][-8:-3] + 'txt', np.array(R_pr, dtype='float32'))
                        # np.savetxt(backupdir + '/test/pr/t_' + valid_files[count][-8:-3] + 'txt', np.array(t_pr, dtype='float32'))
                        # np.savetxt(backupdir + '/test/gt/corners_' + valid_files[count][-8:-3] + 'txt', np.array(corners2D_gt, dtype='float32'))
                        # np.savetxt(backupdir + '/test/pr/corners_' + valid_files[count][-8:-3] + 'txt', np.array(corners2D_pr, dtype='float32'))


            t5 = time.time()

    # Compute 2D projection error, 6D pose error, 5cm5degree error
    px_threshold = 5 # 5 pixel threshold for 2D reprojection error is standard in recent sota 6D object pose estimation works
    eps          = 1e-5
    acc          = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
    acc5cm5deg   = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (len(errs_trans)+eps)
    acc3d10      = len(np.where(np.array(errs_3d) <= diam * 0.1)[0]) * 100. / (len(errs_3d)+eps)
    acc5cm5deg   = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (len(errs_trans)+eps)
    corner_acc   = len(np.where(np.array(errs_corner2D) <= px_threshold)[0]) * 100. / (len(errs_corner2D)+eps)
    mean_err_2d  = np.mean(errs_2d)
    mean_corner_err_2d = np.mean(errs_corner2D)
    nts = float(testing_samples)

    if testtime:
        print('-----------------------------------')
        print('  tensor to cuda : %f' % (t2 - t1))
        print('    forward pass : %f' % (t3 - t2))
        print('get_region_boxes : %f' % (t4 - t3))
        print(' prediction time : %f' % (t4 - t1))
        print('            eval : %f' % (t5 - t4))
        print('-----------------------------------')

    # Print test statistics
    logging('Results of {}'.format(name))
    logging('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))
    logging('   Acc using 10% threshold - {} vx 3D Transformation = {:.2f}%'.format(diam * 0.1, acc3d10))
    logging('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
    logging("   Mean 2D pixel error is %f, Mean vertex error is %f, mean corner error is %f" % (mean_err_2d, np.mean(errs_3d), mean_corner_err_2d))
    logging('   Translation error: %f m, angle error: %f degree, pixel error: % f pix' % (testing_error_trans/nts, testing_error_angle/nts, testing_error_pixel/nts) )

    result_data = {
        'model': modelcfg[23:-4],
        'object': datacfg[14:-5],
        '2d_projection': acc,
        '3d_transformation': acc3d10,
    }

    csv_output_name = 'valid_metrics_distilling.csv' if distiling else 'valid_metrics.csv'

    try:
        df = pd.read_csv(csv_output_name)
        df = df.append(result_data, ignore_index=True)
        df.to_csv(csv_output_name, index=False)
    except:
        df = pd.DataFrame.from_records([result_data])
        df.to_csv(csv_output_name, index=False)
        # shutil.copy2('%s/model.weights' % (backupdir), '%s/model_backup.weights' % (backupdir))


    result_data = {
        'model': modelcfg,
        'acc': acc,
        'acc3d10': acc3d10,
        'acc5cm5deg': acc5cm5deg,
        'mean_err_2d': mean_err_2d,
        'errs_3d': np.mean(errs_3d),
        'mean_corner_err_2d': mean_corner_err_2d,
        'translation_err': testing_error_trans/nts,
        'angle_err': testing_error_angle/nts,
        'px_err': testing_error_pixel/nts
    }

    print(result_data)

    try:
        df = pd.read_csv('test_metrics.csv')
        df = df.append(result_data, ignore_index=True)
        df.to_csv('test_metrics.csv', index=False)
    except:
        df = pd.DataFrame.from_records([result_data])
        df.to_csv('test_metrics.csv', index=False)

    if save:
        predfile = backupdir + '/predictions_linemod_' + name +  '.mat'
        scipy.io.savemat(predfile, {'R_gts': gts_rot, 't_gts':gts_trans, 'corner_gts': gts_corners2D, 'R_prs': preds_rot, 't_prs':preds_trans, 'corner_prs': preds_corners2D})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SingleShotPose')
    parser.add_argument('--datacfg', type=str, default='objects_cfg/txonigiri-test.data') # data config
    parser.add_argument('--modelcfg', type=str, default='models_cfg/tekin/yolo-pose.cfg') # network config
    parser.add_argument('--backupdir', type=str, default='backup/txonigiri') # model backup path
    parser.add_argument('--distiled', type=int, default=0) # if the input model is distiled or not
    args                = parser.parse_args()
    datacfg             = args.datacfg
    modelcfg            = args.modelcfg
    backupdir           = args.backupdir
    distiling           = bool(args.distiled)

    print("configuration file loaded")
    valid(datacfg, modelcfg)

    # txonigiri trained weight #v3.2(95.24%) < v4.1(95.87%) < v5.1(96.75%) < v4.2(97.14%) == v4.3