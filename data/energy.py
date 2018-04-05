import timeit
import numpy as np
# from ..image_util import gaussian_blur_gray_image_nz
# from ..image_util import luminance
import image_util
# from .prob_abs_r import ProbAbsoluteReflectance
# from .prob_abs_s import ProbAbsoluteShading
import sys
from scipy import misc
import math

class IntrinsicEnergy(object):

    def __init__(self, input, params):
        self.input = input
        self.params = params
        # self.prob_abs_r = ProbAbsoluteReflectance(params)
        # self.prob_abs_s = ProbAbsoluteShading(params)

  
    # THIS IS THE MOST IMPORTANT FUNCTIONS IN ZHENGQI PROJECT
    def get_R_features_batch(self):
        final_features = None
        rows = self.input.shape[1]
        cols = self.input.shape[2]

        mask_nz = np.nonzero(np.ones((rows, cols), dtype=bool))
        mask_nnz = mask_nz[0].size
        diag = math.sqrt(rows ** 2 + cols ** 2)

        for i in range(self.input.shape[0]): 
            features = np.zeros((mask_nnz, 5), dtype=np.float32)
            img_rgb = self.input[i,:,:,:]
            img_irg = image_util.rgb_to_irg(img_rgb)

            features[:, 0] = np.reshape(img_irg[:,:,0] / self.params.theta_l, -1)
            features[:, 1] = np.reshape(img_irg[:,:,1] / self.params.theta_c, -1)
            features[:, 2] = np.reshape(img_irg[:,:,2] / self.params.theta_c, -1)

            # pixel location
            features[:, 3] = (
                mask_nz[0] / (self.params.theta_p * diag))
            features[:, 4] = (
                mask_nz[1] / (self.params.theta_p * diag))

            if i == 0:
                final_features = features
            else:
                final_features = np.vstack((final_features, features))

        return final_features
