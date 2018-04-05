learn-reflectance
===========================

implements the intrinsic image decomposition algorithm described in "Learning Intrinsic Image Decomposition from Watching the World, Z. Li and N. Snavely, CVPR 2018". The code is based on "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix". If you use our code for academic purposes, please consider citing:

@inProceedings{BigTimeLi18,
  title={Learning Intrinsic Image Decomposition from Watching the World},
  author={Zhengqi Li and Noah Snavely},e c
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}


#### Dependencies & Compilation:
* The code was written in Pytorch and Python 2 version, but it should be easy to adapt it to Python 3 version needed.
* The sparse matrix construction for spatial-temporal densely connected smoothness term is based on the modifed code from https://github.com/seanbell/intrinsic. In particular, Before trainning the networks, you need to build the C++ code in "data/krahenbuhl2013/". If on Ubuntu and you have installed Eigen3 to its default directory (/usr/include/eigen3), then you can build the C++ code with:
    cd krahenbuhl2013/
    make

Please see https://github.com/seanbell/intrinsic for detail of code dependencies.

#### Evaluation on the IIW/SAW test splits:
* Download IIW and SAW datasets from http://opensurfaces.cs.cornell.edu/publications/intrinsic/#download and https://github.com/kovibalu/saw_release.
* Download pretrained model from and put it in "checkpoints/test_local/paper_final_net_G.pth"
* Change to "self.isTrain = False" in python file "/options/train_options.py"
* To run evaluation on IIW test split, in main direcotry, change the path variable "full_root" the path of IIW dataset in "test_iiw.py" and run:
    python test_iiw.py
* To run evaluation on SAW test split, in main direcotry, change the path variable "saw_root" to the path of SAW dataset in "test_saw.py" and run:
    python test_saw.py


#### Trainning on the BigTime dataset:
* Download the BigTime dataset from our website: http://landmark.cs.cornell.edu/projects/bigtime/BigTime_v1.tar.gz 
* Change path variable "saw_root" to the path of SAW dataset; Change path variable "IIW_root" to the path of IIW dataset; Chnage path variable "train_root" to the path of BigTime.
* build C++ code in "data/krahenbuhl2013/"
* Change to "self.isTrain = True" in python file "/options/train_options.py", and run:
    python train.py

 