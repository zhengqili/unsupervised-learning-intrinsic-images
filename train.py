import time
import torch
import sys
import numpy as np
import random
import os

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from data.data_loader import CreateDataLoaderIIWTest
from models.models import create_model

import torch

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

root = "/"
saw_root = root + "/phoenix/S6/zl548/SAW/saw_release/"

IIW_root = root +'/phoenix/S6/zl548/IIW/iiw-dataset/data/'
IIW_test_list_dir = './IIW_test_list/'

train_root = root + '/phoenix/S6/zl548/AMOS/test/'
train_list_dir = './BigTime_train_list/'
data_loader = CreateDataLoader(train_root, train_list_dir)

dataset = data_loader.load_data()
dataset_size = len(data_loader)


num_iterations = dataset_size/4
model = create_model(opt)
model.switch_to_train()



def validation_iiw(model, list_name):
    total_loss =0.0
    total_loss_eq =0.0
    total_loss_ineq =0.0
    total_count = 0.0

    model.switch_to_eval()

    for j in range(2,3):
        print("============================= Validation IIW MODE ============================", j)

        data_loader_IIW_TEST = CreateDataLoaderIIWTest(IIW_root, IIW_test_list_dir, j)
        dataset_iiw_test = data_loader_IIW_TEST.load_data()

        for i, data in enumerate(dataset_iiw_test):
            stacked_img = data['img_1']
            targets = data['target_1']
            total_whdr, count = model.evlaute_iiw(stacked_img, targets)
            total_loss += total_whdr

            total_count +=count

    model.switch_to_train()

    return total_loss/(total_count)


def validation_SAW(model):
    # parameters for SAW 
    pixel_labels_dir = saw_root + 'saw/saw_pixel_labels/saw_data-filter_size_0-ignore_border_0.05-normal_gradmag_thres_1.5-depth_gradmag_thres_2.0'
    splits_dir = saw_root + 'saw/saw_splits'
    class_weights = [1, 1, 2]
    bl_filter_size = 10
    dataset_split = 'V'

    print("============================= Validation ON SAW============================")
    model.switch_to_eval()
    AP = model.compute_pr(pixel_labels_dir, splits_dir,
                dataset_split, class_weights, bl_filter_size)
    model.switch_to_train()

    print("SAW test AP: %f"%AP)
    return AP


total_steps = 0
best_loss = 100 #validation_iiw(model, 'test_list/')
print(best_loss)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
best_epoch =0
count = 0
num_scale =4 
os_t =0

print('#training images = %d' % dataset_size)

for epoch in range(0, 24):

    if epoch > 0 and epoch % 8 ==0:
        model.scaled_learning_rate(rate = 2.0)

    for i, data in enumerate(dataset):
        print('epoch %d, current iteration %d, best_score %f num_iterations %d best_epoch %d' % (epoch, i, best_loss, num_iterations, best_epoch) )
        stacked_img = data['img_1']
        targets = data['target_1']

        data_set_name = "TL"

        model.set_input(stacked_img, targets)
        model.optimize_parameters(epoch, data_set_name)


    print("================== SAVE AT CURRENT EPOCHS =============================")
    model.save('full_latest')


print("we are done!!!!!")
