Learning Intrinsic Image Decomposition from Watching the World
===========================

This is an implementation of the intrinsic image decomposition algorithm described in "Learning Intrinsic Image Decomposition from Watching the World, Z. Li and N. Snavely, CVPR 2018". The code skeleton is based on "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix". If you use our code for academic purposes, please consider citing:

    @inproceedings{BigTimeLi18,
	  	title={Learning Intrinsic Image Decomposition from Watching the World},
	  	author={Zhengqi Li and Noah Snavely},
	  	booktitle={Computer Vision and Pattern Recognition (CVPR)},
	  	year={2018}
	}

Website: http://www.cs.cornell.edu/projects/bigtime/


#### Dependencies & Compilation:
* The code was written in Pytorch <=0.2 and Python 2, but it should be easy to adapt it to Python 3 version if needed.
* The sparse matrix construction for spatial-temporal densely connected smoothness term is based on the modifed code from https://github.com/seanbell/intrinsic. In particular, you need to build C++ code in "data/krahenbuhl2013/" before trainning the networks. On Ubuntu 16.04 you need to install Eigen3 to its default directory (/usr/include/eigen3), then you can build the C++ code with:
```bash
    cd data/krahenbuhl2013/
    make
```

Please see https://github.com/seanbell/intrinsic for detail.

#### UPDATES: EASY WAY to get predictions/evaluations on the IIW/SAW test sets:
Now we provide precomputed predictions on IIW test set and SAW test set.
* You need to download precomputed predictions for IIW test set in hdf5 format in http://www.cs.cornell.edu/projects/megadepth/dataset/bigtime/bigtime_iiw.zip
* To get evalution results on IIW test set, download IIW dataset and run
```bash
    python compute_iiw_whdr.py
```
(you might need to change judgement_path in this python script to fit to your IIW data path)
* You need to download precomputed predictions for SAW test set in hdf5 format in 
http://www.cs.cornell.edu/projects/megadepth/dataset/bigtime/bigtime_saw.zip
* To get evalution results on SAW test set, download SAW dataset and run
```bash
    python compute_saw_ap.py
```
You need modify 'full_root' in this script and to point to the SAW directory you download. To evlaute on AP% described in the paper, set 'mode = 0' in compute_saw_ap.py.


#### Evaluation on the IIW/SAW test splits:
* Download IIW and SAW datasets from http://opensurfaces.cs.cornell.edu/publications/intrinsic/#download and https://github.com/kovibalu/saw_release.
* Download pretrained model from http://landmark.cs.cornell.edu/projects/bigtime/paper_final_net_G.pth and put it in "checkpoints/test_local/paper_final_net_G.pth"
* Change to "self.isTrain = False" in python file "/options/train_options.py"
* To run evaluation on IIW test split, in main direcotry, change the path variable "full_root" the path of IIW dataset in "test_iiw.py" and run:
```bash
    python test_iiw.py
```
* To run evaluation on SAW test split, in main direcotry, change the path variable "saw_root" to the path of SAW dataset in "test_saw.py" and run:
```bash
    python test_saw.py
```
 
