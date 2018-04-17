    ################################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
################################################################################
import h5py
import torch.utils.data as data
import pickle
import numpy as np
import torch
import os
import os.path
import sys
import math, random
import skimage
from skimage.transform import resize
from skimage import io
from skimage.transform import rotate


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def rgb_to_chromaticity(rgb):
    """ converts rgb to chromaticity """
    irg = np.zeros_like(rgb)
    s = np.sum(rgb, axis=-1) + 1e-8

    irg[..., 0] = rgb[..., 0] / s
    irg[..., 1] = rgb[..., 1] / s
    irg[..., 2] = rgb[..., 2] / s

    return irg

def make_dataset(list_dir):
    file_name = list_dir + "img_batch.p"
    images_list = pickle.load( open( file_name, "rb" ) )
    return images_list

# This is Image loader for unlabel video clips
class ImageFolder(data.Dataset):

    def __init__(self, root, list_dir, transform=None, 
                 loader=None):
        # load image list from hdf5
        img_list = make_dataset(list_dir)
        if len(img_list) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.list_dir = list_dir
        self.img_list = img_list
        self.transform = transform
        x = np.arange(-1, 2)
        y = np.arange(-1, 2)
        self.X, self.Y = np.meshgrid(x, y)
        self.num_scale  = 4
        self.sigma_chro = 0.025
        self.sigma_I = 0.12
        self.half_window = 1
        self.rotation_range  = 5
        self.input_height = 256
        self.input_width = 384

    def construst_sub_L(self, L):
        h = L.shape[0]
        w = L.shape[1]
        sub_L = np.zeros( (9 ,L.shape[0]-2,L.shape[1]-2))

        ct_idx = 0
        for k in range(0, self.half_window*2+1):
            for l in range(0,self.half_window*2+1):
                sub_L[ct_idx,:,:] = L[self.half_window + self.Y[k,l]:h- self.half_window + self.Y[k,l], \
                self.half_window + self.X[k,l]: w-self.half_window + self.X[k,l]] 
                ct_idx += 1

        return sub_L

    def construst_sub_C(self, C):
        h = C.shape[0]
        w = C.shape[1]

        sub_C = np.zeros( (9 ,C.shape[0]-2,C.shape[1]-2, 2))
        ct_idx = 0
        for k in range(0, self.half_window*2+1):
            for l in range(0,self.half_window*2+1):
                sub_C[ct_idx,:,:,:] = C[self.half_window + self.Y[k,l]:h- self.half_window + self.Y[k,l], \
                self.half_window + self.X[k,l]: w-self.half_window + self.X[k,l] , 0:2] 
                ct_idx += 1

        return sub_C


    def construst_R_weights(self, N_c_0, N_L_0):

        center_c = np.repeat( np.expand_dims(N_c_0[4, :, :,:], axis =0), 9, axis = 0)
        center_I = np.repeat( np.expand_dims(N_L_0[4, :, :], axis =0), 9, axis = 0)

        chro_diff = center_c - N_c_0
        I_diff = center_I - N_L_0

        r_w = np.exp( - np.sum( chro_diff**2  , 3) / (self.sigma_chro**2)) * np.exp(- (I_diff**2) /(self.sigma_I**2) )
        
        return r_w

    def DA(self, img, mode, random_pos, random_filp, random_angle, input_height, input_width):

        if random_filp > 0.5:
            img = np.fliplr(img)

        img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3], :]

        img = rotate(img, random_angle, order = mode)
        img = resize(img, (input_height, input_width), order = mode)

        return img

    def generate_random_parameters(self, img):
        original_h, original_w = img.shape[0], img.shape[1]

        random_angle = random.random() * self.rotation_range * 2.0 - self.rotation_range # random angle between -5 --- 5 degree
        random_filp = random.random()

        ratio = float(original_w)/float(original_h)
        random_resize = random.random()

        if ratio > 1.4142:
            random_start_y = random.randint(0, 9) 
            random_start_x = random.randint(0, 29) 
            random_pos = [random_start_y, random_start_y + original_h - 10, random_start_x, random_start_x + original_w - 30]
            input_height, input_width = 256, 384
        elif ratio < 1./1.4142:
            random_start_y = random.randint(0, 29) 
            random_start_x = random.randint(0, 9) 
            random_pos = [random_start_y, random_start_y + original_h - 30, random_start_x, random_start_x + original_w - 10]
            input_height, input_width = 384, 256
        elif ratio > 1.2247:
            random_start_y = random.randint(0, 29) 
            random_start_x = random.randint(0, 9) 
            random_pos = [random_start_y, random_start_y + original_h - 30, random_start_x, random_start_x + original_w - 10]
            input_height, input_width = 256, 384
        elif ratio < 1./ 1.2247:
            random_start_y = random.randint(0, 9) 
            random_start_x = random.randint(0, 29) 
            random_pos = [random_start_y, random_start_y + original_h - 10, random_start_x, random_start_x + original_w - 30]
            input_height, input_width = 384, 256
        else:
            random_start_y = random.randint(0, 9) 
            random_start_x = random.randint(0, 9) 
            random_pos = [random_start_y, random_start_y + original_h - 10, random_start_x, random_start_x + original_w - 10]
            input_height, input_width = 256, 256

        return random_angle, random_filp, random_pos, input_height, input_width

    def __getitem__(self, index):

        targets_1 = {}
        targets_1['path'] = []
        sparse_path_1s = []
        sparse_path_1r = []
        temp_targets = {}

        path_list = self.img_list[index]
        folder_id = path_list[0].split('/')[0]
        #   number of images in one sequence 
        num_imgs = len(path_list)
        num_channel = (self.half_window*2+1)**2;

        # sample image
        img_name = path_list[0].split('/')[-1]
        img_path = self.root + str(folder_id) + "/data/" + img_name
        # load original image 
        srgb_img = np.float32(io.imread(img_path))/ 255.0

        original_h, original_w = srgb_img.shape[0], srgb_img.shape[1]

        random_angle, random_filp, random_pos, input_height, input_width = self.generate_random_parameters(srgb_img)

        # image intensity profiles across the sequence 
        local_intensity_profiles = [None, None, None,None]
        local_intensity_profiles[0] = np.zeros( (num_imgs, num_channel, input_height-self.half_window*2, input_width-self.half_window*2) )
        local_intensity_profiles[1] = np.zeros( (num_imgs, num_channel, input_height/2-self.half_window*2, input_width/2-self.half_window*2) )
        local_intensity_profiles[2] = np.zeros( (num_imgs, num_channel, input_height/4-self.half_window*2, input_width/4-self.half_window*2) )
        local_intensity_profiles[3] = np.zeros( (num_imgs, num_channel, input_height/8-self.half_window*2, input_width/8-self.half_window*2) )

        # random permutation 
        #random_image_list = np.random.permutation(num_imgs)

        # for each image in the sequence 
        for i in range(num_imgs):
            
            img_name = path_list[i].split('/')[-1]
            img_path = self.root + str(folder_id) + "/data/" + img_name
            # load original image 
            srgb_img = np.float32(io.imread(img_path))/ 255.0

            mask_path = self.root + str(folder_id) + "/data/" + img_name[:-4] + "_mask.png"
            # load mask

            mask = np.float32(io.imread(mask_path))/ 255.0
            mask = np.expand_dims(mask, axis = 2)
            mask = np.repeat(mask, 3, axis= 2)

            # do data augmentation 
            assert(mask.shape[0] == srgb_img.shape[0])
            assert(mask.shape[1] == srgb_img.shape[1])

            srgb_img = self.DA(srgb_img, 1,  random_pos, random_filp, random_angle, input_height, input_width)
            mask = self.DA(mask, 0,  random_pos, random_filp, random_angle, input_height, input_width)
            # sky_mask = self.DA(sky_mask, 0,  random_pos, random_filp, random_angle, input_height, input_width)

            srgb_img[srgb_img < 1e-4] = 1e-4
            rgb_img = srgb_img**2.2
            rgb_img[rgb_img < 1e-4] = 1e-4
            chromaticity = rgb_to_chromaticity(rgb_img)
            L0 = np.mean(rgb_img, 2)

            for l in range(self.num_scale):
                N_c_0 = self.construst_sub_C(chromaticity)
                N_L_0 = self.construst_sub_L(L0)
                r_w_s= self.construst_R_weights(N_c_0, N_L_0)

                if ('r_w_s'+ str(l)) not in targets_1:
                    targets_1['r_w_s'+ str(l)] = torch.from_numpy(r_w_s).float().unsqueeze(0)
                    targets_1['mask_' + str(l)] = torch.from_numpy(np.transpose(mask, (2, 0, 1))).float().unsqueeze(0)

                else:
                    targets_1['r_w_s'+ str(l)] = torch.cat( ( targets_1['r_w_s'+ str(l)], \
                                                    torch.from_numpy(r_w_s).float().unsqueeze(0)), 0)
                    targets_1['mask_' + str(l)] = torch.cat( (targets_1['mask_' + str(l)], \
                                                    torch.from_numpy(np.transpose(mask, (2, 0, 1))).float().unsqueeze(0)), 0)

                local_intensity_profiles[l][i,:,:,:] = N_L_0
                L0 = L0[::2,::2]
                chromaticity = chromaticity[::2,::2,:]
                mask = mask[::2,::2,:]

            # create mask
            if 'rgb_img' not in targets_1:
                # targets_1['mask_0'] = torch.from_numpy(mask).float().unsqueeze(0)
                targets_1['rgb_img'] = torch.from_numpy( np.transpose(rgb_img, (2, 0, 1)) ).unsqueeze(0).contiguous().float()                
                final_img = torch.from_numpy(np.transpose(srgb_img, (2, 0, 1))).unsqueeze(0).contiguous().float()
            else:
                # targets_1['mask_0'] = torch.cat( (targets_1['mask_0'], torch.from_numpy(mask).float().unsqueeze(0)), 0)
                targets_1['rgb_img'] = torch.cat( (targets_1['rgb_img'], torch.from_numpy(np.transpose(rgb_img, (2, 0, 1))).float().unsqueeze(0)),0)                
                final_img = torch.cat( (final_img, torch.from_numpy(np.transpose(srgb_img, (2, 0, 1))).unsqueeze(0).contiguous().float()),0)


        k1 = 20.0
        k2 = 4.0
        weight = 12.0
        offset = 1.0/weight
        
        # compute median of Intensity profiles for each scale
        for l in range(0, self.num_scale):
            intensity_profiles = local_intensity_profiles[l]
            log_ratio_profiles =  np.log( np.repeat(  np.expand_dims( intensity_profiles[:,4,:,:], 1) , 9, axis = 1)) - np.log(intensity_profiles) 
            median_ratio = np.median(log_ratio_profiles, axis = 0)
            median_ratio = np.repeat( np.expand_dims(median_ratio, 0), num_imgs, axis = 0) 
            relative_changes = (log_ratio_profiles - median_ratio)/(median_ratio + 1e-6)

            sw_1 = np.exp(- k1 * (log_ratio_profiles - median_ratio)**2 ) 
            sw_2 = np.exp(- k2 * (relative_changes)**2 ) 
            shading_w = np.maximum(sw_1, sw_2)

            shading_w =  torch.from_numpy(shading_w).float()
            R_w = targets_1['r_w_s' + str(l)]
            R_w, index = torch.median(R_w, 0)
            R_w = 1 - R_w.unsqueeze(0).repeat(shading_w.size(0), 1,1,1)

            shading_w = torch.mul(offset + R_w, shading_w)

            targets_1['shading_w_'+str(l)] = weight * shading_w 


        return final_img, targets_1, sparse_path_1r


    def __len__(self):
        return len(self.img_list)


class IIW_ImageFolder(data.Dataset):

    def __init__(self, root, list_dir, mode, is_flip, transform=None, 
                 loader=None):
        # load image list from hdf5
        img_list = make_dataset(list_dir)
        if len(img_list) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.list_dir = list_dir

        self.img_list = img_list
        self.transform = transform
        self.loader = loader
        self.num_scale  = 4
        self.sigma_chro = 0.025
        self.sigma_I = 0.1
        self.half_window = 1
        self.current_o_idx = mode
        self.set_o_idx(mode)
        x = np.arange(-1, 2)
        y = np.arange(-1, 2)
        self.X, self.Y = np.meshgrid(x, y)

    def set_o_idx(self, o_idx):
        self.current_o_idx = o_idx

        if o_idx == 0:
            self.height = 256
            self.width = 384
        elif o_idx == 1:
            self.height = 384
            self.width = 256
        elif o_idx == 2:
            self.height = 256
            self.width = 256
        elif o_idx == 3:
            self.height = 384
            self.width = 512
        else:
            self.height = 512
            self.width = 384

    def iiw_loader(self, img_path):
        
        img_path = img_path[-1][:-3]
        img_path = self.root + img_path
        img = np.float32(io.imread(img_path))/ 255.0
        oringinal_shape = img.shape

        img = resize(img, (self.height, self.width))

        return img, oringinal_shape


    def __getitem__(self, index):
        targets_1 = {}

        img_id = self.img_list[self.current_o_idx][index].split('/')[-1][0:-6]
        judgement_path = self.root + img_id + 'json'

        img, oringinal_shape = self.iiw_loader(self.img_list[self.current_o_idx][index].split('/'))

        targets_1['path'] = self.img_list[self.current_o_idx][index]
        targets_1["judgements_path"] = judgement_path
        targets_1["oringinal_shape"] = oringinal_shape

        final_img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2,0,1)))).contiguous().float()

        return final_img, targets_1


    def __len__(self):
        return len(self.img_list[self.current_o_idx])


