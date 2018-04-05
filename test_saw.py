import time
import torch
import numpy as np
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
import sys, traceback
from models.models import create_model


root = '/'
saw_root = root + "/phoenix/S6/zl548/SAW/saw_release/"

dataset_split = 'E' # Test set

model = create_model(opt)


def test_SAW(model):
    # parameters for SAW 
    pixel_labels_dir = saw_root + 'saw/saw_pixel_labels/saw_data-filter_size_0-ignore_border_0.05-normal_gradmag_thres_1.5-depth_gradmag_thres_2.0'
    splits_dir = saw_root + 'saw/saw_splits'
    class_weights = [1, 1, 2]
    bl_filter_size = 10

    print("============================= Validation ON SAW============================")
    model.switch_to_eval()
    AP = model.compute_pr(pixel_labels_dir, splits_dir,
                dataset_split, class_weights, bl_filter_size)

    print("SAW test AP: %f"%AP)
    return AP


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
best_epoch =0


print("WE ARE IN TESTING SAW")
test_SAW(model)
