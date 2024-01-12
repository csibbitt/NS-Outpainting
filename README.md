# WIP
## Migrating this code to Tensor Flow 2

Initial migrations are complete, but many transitional compat APIs and deprecated methods remain.

I'm learning TensorFlow as I go. This code has not yet been fully tested and may not work (Please send GPUs).

I have done a partial training run (to Step ~75,000 before hitting the "NaN Detected" code) and the results are great.

The model is being used as the core of an art project that makes never-ending landscape panoramas.

## Known deviations for original model

I've probably made a mistake or two somewhere, but have tried to keep all the parameters and configuration of the model identical to the research paper, except for the following intentional changes, so far:

* Many blocks (convolution, encoder, decoder, grb, identity) shared a single initializer within the block, leading (I think?) to convolution kernels in the block all being initialized the same. I've changed this so that each convolution layer gets it's own initializer. It's possible the original code worked this same way and the fix is an artifact of porting to V2? I haven't double checked.

# Very Long Natural Scenery Image Prediction by Outpainting (NS-Outpainting)
A neural architecture for scenery image outpaiting ([ICCV 2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_Very_Long_Natural_Scenery_Image_Prediction_by_Outpainting_ICCV_2019_paper.pdf)), implemented in [TensorFlow](http://www.tensorflow.org).

The architecture has an ability to generate a very long high-quality prediction from a small input image by outpaiting:
<img src="https://github.com/z-x-yang/NS-Outpainting/raw/master/examples/3.png" width="90%"/>
<img src="https://github.com/z-x-yang/NS-Outpainting/raw/master/examples/2.png" width="90%"/>
<img src="https://github.com/z-x-yang/NS-Outpainting/raw/master/examples/1.png" width="90%"/>

## Requirements and Preparation

Please install `TensorFlow>=1.3.0`, `Python>=3.6`.

For training and testing, we collect a new outpainting dataset, which has 6,000 images containing complex natural scenes. You can download the raw dataset from [here](https://drive.google.com/file/d/15rGKgeNHWqjs90An7wpZXJMz-zFaC1q0/view?usp=sharing) and split the training and testing set by yourself. Or, you can get our split from [here](https://drive.google.com/file/d/1LDRx0W6zo_eCZwN92pGgGZSCrqzB3KZ6/view?usp=sharing) (TFRecord format, 128 resolution, 5,000 images for training and 1,000 for testing).

## Usage

For training and evaluation, you can use [train.sh](/train.sh) and [eval.sh](/eval.sh). Please remember to set the TFRecord dataset path inside them.

Besides, you can get our **pretrain model** from [here](https://drive.google.com/file/d/1-DLSwNkB93MMKaYVO1rmPP9iJllXDJrg/view?usp=sharing), and run eval_model.py to evaluate it.

After running eval_model.py, the evaluation process will store 4 types of images: 
  1) "ori_xxx.jpg", the groundtruth images of size 128x256; 
  2) "m0_xxx.jpg", the 1-step predictions of size 128x256 without any post-processing methods;
  3) "m1_xxx.jpg", the 1-step predictions of size 128x256 with smoothly stitching;
  4) "endless_xxx.jpg", the 4-step predictions of size 128x640.

Notably, we measure Inception Score and Inception Distance between "ori_xxx.jpg" and "m0_xxx.jpg" in our paper.

## Citation
```
@inproceedings{yang2019very,
  title={Very Long Natural Scenery Image Prediction by Outpainting},
  author={Yang, Zongxin and Dong, Jian and Liu, Ping and Yang, Yi and Yan, Shuicheng},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={10561--10570},
  year={2019}
}
```
