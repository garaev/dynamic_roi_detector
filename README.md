# Dynamic roi detector

This repository contains the source code and experimental results for the dynamic roi detector. Dynamic roi means roi pool or roi align is used depending on the size of the proposal box.

## Requirements
* Ubuntu 20.04 or older

* Python 3.6
  ```
  $ conda create -n dynamic_roi_pooler python=3.6
    conda activate dynamic_roi_pooler
  ```
* torch 1.0 and torchvision 0.2.1
  ```
  $ conda install pytorch==1.0.0 torchvision==0.2.1 -c pytorch
  ```
* tqdm
    ```
    $ pip install tqdm
    ```

* tensorboardX
    ```
    $ pip install tensorboardX
    ```

* OpenCV 3.4 (required by `infer_stream.py`)
    ```
    $ pip install opencv-python~=3.4
    ```

* websockets (required by `infer_websocket.py`)
    ```
    $ pip install websockets
    ```


## Setup

1. Prepare data
    1. For `PASCAL VOC 2007`

        1. Download dataset

            - [Training / Validation](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) (5011 images)
            - [Test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) (4952 images)

        1. Extract to data folder, now your folder structure should be like:

            ```
            easy-faster-rcnn.pytorch
                - data
                    - VOCdevkit
                        - VOC2007
                            - Annotations
                                - 000001.xml
                                - 000002.xml
                                ...
                            - ImageSets
                                - Main
                                    ...
                                    test.txt
                                    ...
                                    trainval.txt
                                    ...
                            - JPEGImages
                                - 000001.jpg
                                - 000002.jpg
                                ...
                    - ...
            ```

1. Build `Non Maximum Suppression` and `ROI Align` modules (modified from [facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark))

    1. Install

        ```
        $ conda install mkl=2018 -c intel
          python support/setup.py develop
        ```

    1. Uninstall

        ```
        $ python support/setup.py develop --uninstall
        ```

    1. Test

        ```
        $ python test/nms/test_nms.py
        ```

        * Result

            ![](images/test_nms.png?raw=true)


## Usage

1. Train

        $ python train.py -s=voc2007 -b=resnet18

1. Evaluate

        $ python eval.py -s=voc2007 -b=resnet18 /path/to/checkpoint.pth


## Results
All experiments done with ResNet18 backbone on one 1080ti with batch 20, average learning time is ~5h. All other info you can check in config. Only successful experiments were included in the table.

  <table>
      <tr>
          <th>Name of experiment</th>
          <th>Description</th>
          <th>Pooler mode</th>
          <th>mAP</th>
          <th>Inference Speed on GPU (ms)</th>
          <th>Checkpoints</th>
      </tr>
      <tr>
          <td>RoI Pool</td>
          <td>RoI pooling for all proposal bboxes.</td>
          <td>POOLING</td>
          <td>0.6235</td>
          <td>54</td>
          <td>
            <a href="https://drive.google.com/file/d/1UxS7X39qsmVRsKLQJH3jMeo983mp0HPF/view?usp=sharing">
                Weights
            </a>
          </td>
      <tr>
          <td>RoI Align</td>
          <td>RoI align for all proposal bboxes.</td>
          <td>ALIGN</td>
          <td>0.6566</td>
          <td>82</td>
          <td>
            <a href="https://drive.google.com/file/d/18SCYAAX7mkuZDCtrpFwwtyzBmeeecdEV/view?usp=sharing">
                Weights
            </a>
          </td>
      <tr>
          <td>Dynamic</td>
          <td>For eash batch of RoI calculating average proposal bbox area. The ROI pool is then used if the area of the ROI is greater than the average and the ROI align is used otherwise.</td>
          <td>DYNAMIC_AVG</td>
          <td>0.6387</td>
          <td>98</td>
          <td>
            <a href="https://drive.google.com/file/d/1HhSdjZcJpcbE7VMEiSlCAI-Z8c4wbG1C/view?usp=sharing">
                Weights
            </a>
          </td>
      </tr>
      <tr>
          <td>Dynamic reverse</td>
          <td>Everything is the same as in the previous line, but now RoI pool is used for small objects, and RoI align is used for large ones.</td>
          <td>DYNAMIC_REVERSE</td>
          <td>0.6240</td>
          <td>89</td>
          <td>
            <a href="https://drive.google.com/file/d/1m2G9ZKA_fnyKacS_sVOMzmDMQ4qAMfEw/view?usp=sharing">
                Weights
            </a>
          </td>
      </tr>
      <tr>
          <td>Dynamic 128x128</td>
          <td>If the proposal bbox area is greater than 128x128, then the RoI pool is used, otherwise the RoI alignment is used. The numbers 128, 256 and 512 are taken as these are the typical sizes of anchors in the RPN. Thus, I test the assumption that different poolers may be suitable for different size groups of RPN outputs.</td>
          <td>DYNAMIC_128x128</td>
          <td>0.6354</td>
          <td>70</td>
          <td>
            <a href="https://drive.google.com/file/d/1VjYQl5iLs7JrRjGV_ZEbiolqds5E1ESE/view?usp=sharing">
                Weights
            </a>
          </td>
      </tr>
      <tr>
          <td>Dynamic 256x256</td>
          <td>If the proposal bbox area is greater than 256x256, then the RoI pool is used, otherwise the RoI alignment is used.</td>
          <td>DYNAMIC_256x256</td>
          <td>0.6290</td>
          <td>74</td>
          <td>
            <a href="https://drive.google.com/file/d/1Z71Y0DMknfG2HjHSf1m8uxAoAmuiqHuZ/view?usp=sharing">
                Weights
            </a>
          </td>
      </tr>
      <tr>
          <td>Dynamic 512x512</td>
          <td>If the proposal bbox area is greater than 512x512, then the RoI pool is used, otherwise the RoI alignment is used.</td>
          <td>DYNAMIC_512x512</td>
          <td>0.6367</td>
          <td>79</td>
          <td>
            <a href="https://drive.google.com/file/d/1JdlX5J_-rOAo9vk7Pslfl0D177y2k0dQ/view?usp=sharing">
                Weights
            </a>
          </td>
      </tr>
  </table>

## Conclusions
1. RoI align works the best, but it is slow enough.
2. RoI align is more useful for small objects, which is confirmed by the accuracy of DYNAMIC_AVG (mAP = 0.6387) and DYNAMIC_REVERSE (mAP = 0.6240).
3. All dynamic pools work rather slowly, since they use a loop in their implementation, so dynamic pools are only beneficial if the RoI align branch is rarely selected. This happens often in DYNAMIC_128x128, so it can be called the best in terms of accuracy to speed. In case you need to get accuracy and speed between RoI pool and RoI align, it is best to use DYNAMIC_128x128.

## Further exploration
1. If you figure out how to do dynamic pooling based on matrix operations without a loop, then this will significantly speed up this approach.

## References
The code for training is taken from this repository: https://github.com/potterhsu/easy-faster-rcnn.pytorch.git
