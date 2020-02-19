# Squeezed Deep 6DoF Object Detection Using Knowledge Distillation
 
This work is a modification of the original forked work to add other architectures and Knowledge Distillation.
The architectures used to change the YOLOv2 of the original work was YOLO-LITE and Tiny-YOLO. This modification reduce the network weights by 99%.
The weights of the networks can be downloaded [here](https://drive.google.com/drive/folders/1FsWjgrqzgwHlQCmApwvnUKOVS-bjatDZ?usp=sharing).

### Introduction

For further questions, consult the original work where it is described in more detail.

To add Knowledge Distillation to training, change the Distillation flag to true in the training code (train.py) and in the Darknet code (darknet.py).

The results obteined to 2D Reprojection metric are:
![image](https://drive.google.com/file/d/1JarpA3X7iVIVxC4bN-lbliGlZeZNbJOg/view)

### License

[MIT](https://choosealicense.com/licenses/mit/)

#### Environment and dependencies

The code is tested on Linux with CUDA v10. The implementation is based on PyTorch 1.2 and tested on Python3.7. The code requires the following dependencies that could be installed with conda or pip: numpy, scipy, PIL, opencv-python.  

#### Downloading and preparing the data

Inside the main code directory, run the following to download and extract (1) the preprocessed LINEMOD dataset, (2) trained models for the LINEMOD dataset, (3) the trained model for the OCCLUSION dataset, (4) background images from the VOC2012 dataset respectively.
```
wget -O LINEMOD.tar --no-check-certificate "https://onedrive.live.com/download?cid=05750EBEE1537631&resid=5750EBEE1537631%21135&authkey=AJRHFmZbcjXxTmI"
wget -O backup.tar --no-check-certificate "https://onedrive.live.com/download?cid=0C78B7DE6C569D7B&resid=C78B7DE6C569D7B%21191&authkey=AP183o4PlczZR78"
wget -O multi_obj_pose_estimation/backup_multi.tar --no-check-certificate  "https://onedrive.live.com/download?cid=05750EBEE1537631&resid=5750EBEE1537631%21136&authkey=AFQv01OSbvhGnoM"
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/darknet19_448.conv.23 -P cfg/
tar xf LINEMOD.tar
tar xf backup.tar
tar xf multi_obj_pose_estimation/backup_multi.tar -C multi_obj_pose_estimation/
tar xf VOCtrainval_11-May-2012.tar
```
Alternatively, you can directly go to the links above and manually download and extract the files at the corresponding directories. The whole download process might take a long while (~60 minutes). Please also be aware that access to OneDrive in some countries might be limited.

#### Training the model

To add Knowledge Distillation to training, change the Distillation flag to true in the training code (train.py) and in the Darknet code (darknet.py).

To train the model run,

```
python train.py --datacfg [path_to_data_config_file] --modelcfg [path_to_model_config_file] --initweightfile [path_to_initialization_weights] --pretrain_num_epochs [number_of_epochs to pretrain]
```
e.g.
```
python train.py --datacfg cfg/ape.data --modelcfg cfg/yolo-pose.cfg --initweightfile cfg/darknet19_448.conv.23 --pretrain_num_epochs 15
```
Start with an already pretrained model on LINEMOD, for faster convergence.

**[datacfg]** contains information about the training/test splits, 3D object models and camera parameters

**[modelcfg]** contains information about the network structure

**[initweightfile]** contains initialization weights.  <<darknet19_448.conv.23>> contains the network weights pretrained on ImageNet. The weights "backup/[OBJECT_NAME]/init.weights" are pretrained on LINEMOD for faster convergence. We found it effective to pretrain the model without confidence estimation first and fine-tune the network later on with confidence estimation as well. "init.weights" contain the weights of these pretrained networks. However, you can also still train the network from a more crude initialization (with weights trained on ImageNet). This usually results in a slower and sometimes slightly worse convergence. You can find in cfg/ folder the file <<darknet19_448.conv.23>> that includes the network weights pretrained on ImageNet.

At the start of the training you will see an output like this:

```
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  32
    1 max          2 x 2 / 2   416 x 416 x  32   ->   208 x 208 x  32
    2 conv     64  3 x 3 / 1   208 x 208 x  32   ->   208 x 208 x  64
    3 max          2 x 2 / 2   208 x 208 x  64   ->   104 x 104 x  64
    ...
   30 conv     20  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x  20
   31 detection
```

This defines the network structure. During training, the best network model is saved into the "model.weights" file. To train networks for other objects, just change the object name while calling the train function, e.g., "python train.py cfg/duck.data cfg/yolo-pose.cfg backup/duck/init.weights". If you come across GPU memory errors while training, you could try lowering the batch size, to for example 16 or 8, to fit into the memory. The open source version of the code has undergone strong refactoring and furthermore some models had to be retrained. The retrained models that we provide do not change much from the initial results that we provide (sometimes slight worse and sometimes slightly better).

#### Testing the model

To test the model run

```
python valid.py --datacfg [path_to_data_config_file] --modelcfg [path_to_model_config_file] --weightfile [path_to_trained_model_weights]
```
e.g.
```
python valid.py --datacfg cfg/ape.data --modelcfg cfg/yolo-pose.cfg --weightfile backup/ape/model_backup.weights
```

You could also use valid.ipynb to test the model and visualize the results.

#### Label files

Our label files consist of 21 ground-truth values. We predict 9 points corresponding to the centroid and corners of the 3D object model. Additionally we predict the class in each cell. That makes 9x2+1 = 19 points. In multi-object training, during training, we assign whichever anchor box has the most similar size to the current object as the responsible one to predict the 2D coordinates for that object. To encode the size of the objects, we have additional 2 numbers for the range in x dimension and y dimension. Therefore, we have 9x2+1+2 = 21 numbers. 
 
Respectively, 21 numbers correspond to the following: 1st number: class label, 2nd number: x0 (x-coordinate of the centroid), 3rd number: y0 (y-coordinate of the centroid), 4th number: x1 (x-coordinate of the first corner), 5th number: y1 (y-coordinate of the first corner), ..., 18th number: x8 (x-coordinate of the eighth corner), 19th number: y8 (y-coordinate of the eighth corner), 20th number: x range, 21st number: y range.
 
The coordinates are normalized by the image width and height: x / image_width and y / image_height. This is useful to have similar output ranges for the coordinate regression and object classification tasks. 
