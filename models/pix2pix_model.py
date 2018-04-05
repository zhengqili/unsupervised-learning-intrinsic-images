import numpy as np
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
import sys, traceback
import h5py
import os.path
import json
from . import saw_utils
from scipy.ndimage.filters import maximum_filter
import skimage
from scipy.ndimage.measurements import label
from skimage.transform import resize

class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def __init__(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)

        # load/define networks
        model = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netG, 'batch', opt.use_dropout, self.gpu_ids)


        if not self.isTrain:
            model_parameters = self.load_network(model, 'G', 'paper_final')
            model.load_state_dict(model_parameters)

        self.netG  = model.cuda()
        self.lr = opt.lr
        self.old_lr = opt.lr

        if self.isTrain:            
            self.netG.train()
            self.criterion_joint = networks.JointLoss() 
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(0.9, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            print('-----------------------------------------------')

    def set_input(self, input, targets):
        self.num_pair = input.size(0)
        self.input.resize_(input.size()).copy_(input)
        self.targets = targets

    def forward(self):
        # print("We are Forwarding !!")
        self.input_images = Variable(self.input.float().cuda(), requires_grad = False)
        self.prediction_S, self.prediction_R, self.rgb_s = self.netG.forward(self.input_images)

    #get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self, epoch, data_set_name):

        self.loss_joint = self.criterion_joint(self.input_images, self.prediction_S, \
                        self.prediction_R, self.rgb_s, self.targets, data_set_name, epoch)
        print("loss is %f "%self.loss_joint)
        self.loss_joint_var = self.criterion_joint.get_loss_var()
        self.loss_joint_var.backward()


    def optimize_parameters(self, epoch, data_set_name):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G(epoch, data_set_name)
        self.optimizer_G.step()


    def evlaute_iiw(self, input_, targets):
        # switch to evaluation mode
        input_images = Variable(input_.cuda() , requires_grad = False)
        prediction_S, prediction_R , rgb_s = self.netG.forward(input_images)

        prediction_R = torch.exp(prediction_R)

        return self.evaluate_WHDR(prediction_R, targets)

    def compute_whdr(self, reflectance, judgements, delta=0.1):
        points = judgements['intrinsic_points']
        comparisons = judgements['intrinsic_comparisons']
        id_to_points = {p['id']: p for p in points}
        rows, cols = reflectance.shape[0:2]
 
        error_sum = 0.0
        weight_sum = 0.0

        for c in comparisons:
            # "darker" is "J_i" in our paper
            darker = c['darker']
            if darker not in ('1', '2', 'E'):
                continue

            # "darker_score" is "w_i" in our paper
            weight = c['darker_score']
            if weight <= 0.0 or weight is None:
                continue

            point1 = id_to_points[c['point1']]
            point2 = id_to_points[c['point2']]
            if not point1['opaque'] or not point2['opaque']:
                continue

            # convert to grayscale and threshold
            l1 = max(1e-10, np.mean(reflectance[
                int(point1['y'] * rows), int(point1['x'] * cols), ...]))
            l2 = max(1e-10, np.mean(reflectance[
                int(point2['y'] * rows), int(point2['x'] * cols), ...]))

            # # convert algorithm value to the same units as human judgements
            if l2 / l1 > 1.0 + delta:
                alg_darker = '1'
            elif l1 / l2 > 1.0 + delta:
                alg_darker = '2'
            else:
                alg_darker = 'E'


            if darker != alg_darker:
                error_sum += weight

            weight_sum += weight

        if weight_sum:
            return (error_sum / weight_sum)
        else:
            return None

    def evaluate_WHDR(self, prediction_R, targets):
        total_whdr = float(0)
        count = float(0) 

        for i in range(0, prediction_R.size(0)):
            print(targets['path'][i])
            prediction_R_np = prediction_R.data[i,:,:,:].cpu().numpy()
            prediction_R_np = np.transpose(prediction_R_np, (1,2,0))

            o_h = targets['oringinal_shape'][0].numpy()
            o_w = targets['oringinal_shape'][1].numpy()
            # resize to original resolution 
            prediction_R_np = resize(prediction_R_np, (o_h[i],o_w[i]), order=1, preserve_range=True)
            # load Json judgement 
            judgements = json.load(open(targets["judgements_path"][i]))
            whdr = self.compute_whdr(prediction_R_np, judgements, 0.1)

            total_whdr += whdr
            count += 1.

        return total_whdr, count


    def switch_to_train(self):
        self.netG.train()

    def switch_to_eval(self):
        self.netG.eval()

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        # self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        self.lr = lr

        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def scaled_learning_rate(self, rate = 2.0):
        lr = self.old_lr /rate
        self.lr = lr

        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


    def compute_pr(self, pixel_labels_dir, splits_dir, dataset_split, class_weights, bl_filter_size, thres_count=400):

        thres_list = saw_utils.gen_pr_thres_list(thres_count)
        photo_ids = saw_utils.load_photo_ids_for_split(
            splits_dir=splits_dir, dataset_split=dataset_split)

        plot_arrs = []
        line_names = []

        fn = 'pr-%s' % {'R': 'train', 'V': 'val', 'E': 'test'}[dataset_split]
        title = '%s Precision-Recall' % (
            {'R': 'Training', 'V': 'Validation', 'E': 'Test'}[dataset_split],
        )

        print("FN ", fn)
        print("title ", title)

        # compute PR 
        rdic_list = self.get_precision_recall_list_new(pixel_labels_dir=pixel_labels_dir, thres_list=thres_list,
            photo_ids=photo_ids, class_weights=class_weights, bl_filter_size = bl_filter_size)

        plot_arr = np.empty((len(rdic_list) + 2, 2))

        # extrapolate starting point 
        plot_arr[0, 0] = 0.0
        plot_arr[0, 1] = rdic_list[0]['overall_prec']

        for i, rdic in enumerate(rdic_list):
            plot_arr[i+1, 0] = rdic['overall_recall']
            plot_arr[i+1, 1] = rdic['overall_prec']

        # extrapolate end point
        plot_arr[-1, 0] = 1
        plot_arr[-1, 1] = 0.5

        AP = np.trapz(plot_arr[:,1], plot_arr[:,0])

        return AP


    def get_precision_recall_list_new(self, pixel_labels_dir, thres_list, photo_ids,
                                  class_weights, bl_filter_size):

        output_count = len(thres_list)
        overall_conf_mx_list = [
            np.zeros((3, 2), dtype=int)
            for _ in xrange(output_count)
        ]

        count = 0 
        total_num_img = len(photo_ids)

        for photo_id in (photo_ids):
            print("photo_id ", count, photo_id, total_num_img)

            saw_img = saw_utils.load_img_arr(photo_id)
            original_h, original_w = saw_img.shape[0], saw_img.shape[1]
            saw_img = saw_utils.resize_img_arr(saw_img)

            saw_img = np.transpose(saw_img, (2,0,1))
            input_ = torch.from_numpy(saw_img).unsqueeze(0).contiguous().float()
            input_images = Variable(input_.cuda() , requires_grad = False)

            prediction_S, prediction_R, rgb_s = self.netG.forward(input_images) 

            prediction_Sr = torch.exp(prediction_S)
            prediction_S_np = prediction_Sr.data[0,0,:,:].cpu().numpy() 
            prediction_S_np = resize(prediction_S_np, (original_h, original_w), order=1, preserve_range=True)

            # compute confusion matrix
            conf_mx_list = self.eval_on_images( shading_image_arr = prediction_S_np,
                pixel_labels_dir=pixel_labels_dir, thres_list=thres_list,
                photo_id=photo_id, bl_filter_size = bl_filter_size
            )


            # if np.all(conf_mx_list == 0) == False:
            #     print("no label")
            #     continue

            for i, conf_mx in enumerate(conf_mx_list):
                # If this image didn't have any labels
                if conf_mx is None:
                    continue

                overall_conf_mx_list[i] += conf_mx

            count += 1

        ret = []
        for i in xrange(output_count):
            overall_prec, overall_recall = saw_utils.get_pr_from_conf_mx(
                conf_mx=overall_conf_mx_list[i], class_weights=class_weights,
            )

            ret.append(dict(
                overall_prec=overall_prec,
                overall_recall=overall_recall,
                overall_conf_mx=overall_conf_mx_list[i],
            ))

        return ret


    def eval_on_images(self, shading_image_arr, pixel_labels_dir, thres_list, photo_id, bl_filter_size):
        """
        This method generates a list of precision-recall pairs and confusion
        matrices for each threshold provided in ``thres_list`` for a specific
        photo.

        :param shading_image_arr: predicted shading images

        :param pixel_labels_dir: Directory which contains the SAW pixel labels for each photo.

        :param thres_list: List of shading gradient magnitude thresholds we use to
        generate points on the precision-recall curve.

        :param photo_id: ID of the photo we want to evaluate on.

        :param bl_filter_size: The size of the maximum filter used on the shading
        gradient magnitude image. We used 10 in the paper. If 0, we do not filter.
        """

        shading_image_grayscale = shading_image_arr
        shading_image_grayscale[shading_image_grayscale < 1e-4] = 1e-4
        shading_image_grayscale = np.log(shading_image_grayscale)

        shading_gradmag = saw_utils.compute_gradmag(shading_image_grayscale)
        shading_gradmag = np.abs(shading_gradmag)

        if bl_filter_size:
            shading_gradmag_max = maximum_filter(shading_gradmag, size=bl_filter_size)

        # We have the following ground truth labels:
        # (0) normal/depth discontinuity non-smooth shading (NS-ND)
        # (1) shadow boundary non-smooth shading (NS-SB)
        # (2) smooth shading (S)
        # (100) no data, ignored
        y_true = saw_utils.load_pixel_labels(pixel_labels_dir=pixel_labels_dir, photo_id=photo_id)
        
        y_true = np.ravel(y_true)
        ignored_mask = y_true == 100

        # If we don't have labels for this photo (so everything is ignored), return
        # None
        if np.all(ignored_mask):
            print("no labels")
            return [None] * len(thres_list)

        ret = []
        for thres in thres_list:
            y_pred = (shading_gradmag < thres).astype(int)
            y_pred_max = (shading_gradmag_max < thres).astype(int)
            y_pred = np.ravel(y_pred)
            y_pred_max = np.ravel(y_pred_max)
            # Note: y_pred should have the same image resolution as y_true
            assert y_pred.shape == y_true.shape

            confusion_matrix = saw_utils.grouped_confusion_matrix(y_true[~ignored_mask], y_pred[~ignored_mask], y_pred_max[~ignored_mask])
            ret.append(confusion_matrix)

        return ret

