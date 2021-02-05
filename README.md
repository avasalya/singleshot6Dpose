# Introduction
##  without Distillation
This repository is adapted from the original implementation [Squeezed Deep 6DoF Object Detection Using Knowledge Distillation](https://arxiv.org/abs/2003.13586).

This is a forked version of the original work (https://github.com/heitorcfelix/singleshot6Dpose) but without Knowledge Distillation.

## License
[MIT](https://choosealicense.com/licenses/mit/)

<br/>

# Create your own custom dataset

## Label files

 Label files consist of 21 ground-truth values. We predict 9 points corresponding to the centroid and corners of the 3D object model. Additionally we predict the class in each cell. That makes 9x2+1 = 19 points. In multi-object training, during training, we assign whichever anchor box has the most similar size to the current object as the responsible one to predict the 2D coordinates for that object. To encode the size of the objects, we have additional 2 numbers for the range in x dimension and y dimension. Therefore, we have 9x2+1+2 = 21 numbers.

Respectively, 21 numbers correspond to the following: 1st number: class label, 2nd number: x0 (x-coordinate of the centroid), 3rd number: y0 (y-coordinate of the centroid), 4th number: x1 (x-coordinate of the first corner), 5th number: y1 (y-coordinate of the first corner), ..., 18th number: x8 (x-coordinate of the eighth corner), 19th number: y8 (y-coordinate of the eighth corner), 20th number: x range, 21st number: y range.

The coordinates are normalized by the image width and height: x / image_width and y / image_height. This is useful to have similar output ranges for the coordinate regression and object classification tasks.

* use this repository to create your own dataset for Yolo6D (develop branch) https://github.com/avasalya/RapidPoseLabels/tree/develop (instructions are provided)

   * I have made changes in the original repository to meet the necessary requirements to produce dataset for yolo6D.

  * please read `dataset.sh` for further instructions on how to create your own dataset

  * once you run `dataset.sh`, this will generate the `out_cur_date` folder with several contents, however to train `Yolo6D` you just need to copy `rgb`, `mask`, `label` folders, and `train.txt`, `test.txt`, `yourObject.ply` files.

  * Please refer to original work here for further support, https://github.com/rohanpsingh/RapidPoseLabels, PS: but my forked branch is not updated.



<br>

# Training the model

* Download `'VOCdevkit/VOC2012/JPEGImages` and place it in the main directory, used to replace background images while training.
   * wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
   * tar xf VOCtrainval_11-May-2012.tar

* place the copied `rgb`, `mask`, `label` folders and `train.txt`, `test.txt` files to the folder `linemod/customObj/data/01/`

* create your own custom `custom.data` file inside `objects_cfg` folder.

* place the `yourObject.ply` into `linemod/txonigiri/models/`

* create conda environment `conda env create -f environment-train.yml`
  <!-- * install following lib manually
  `open3d`,
  `rospkg`,
  `chainer_mask_rcn`,
  `pyrealsense2` -->


* To train the model run,
  ```
  python train.py --datacfg [path_to_data_config_file (objects_cfg/custom.data)] --modelcfg [path_to_model_config_file (models_cfg/tekin/yolo-pose.cfg)] --initweightfile [path_to_initialization_weights (cfg/darknet19_448.conv.23)] --backupdir [backup/create_your_custom_dir]
  ```
  * e.g.
    ```
    python train.py --datacfg cfg/onigiri-tx.data --modelcfg cfg/yolo-pose.cfg --initweightfile cfg/darknet19_448.conv.23 --backupdir backup/txonigiri
    ```

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

<br>

# Testing the model

* Use `valid-wt-accuracy.py` to test the model and visualize the results.

  * modify the path to your `custom-test.cfg` file
    * e.g. `datacfg = 'objects_cfg/onigiri-test.data'`

  * modify following as per your requirements
    * `save            = False`
    * `testtime        = False`
    * `visualize       = True`

  * finally run `python3 valid-wt-accuracy.py`



<br>

# Citation
If you use this in your research, please cite the original project.
```
@article{felix2020squeezed6dof,
  title={Squeezed Deep 6DoF Object Detection Using Knowledge Distillation},
  author={Felix, Heitor and Rodrigues, Walber M and Mac{\^e}do, David and Sim{\~o}es, Francisco and Oliveira, Adriano LI and Teichrieb, Veronica and Zanchettin, Cleber},
  journal={arXiv preprint arXiv:2003.13586},
  year={2020}
}
```
