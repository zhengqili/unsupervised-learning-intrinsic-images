import random
import numpy as np
import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.image_folder import *
from builtins import object
import sys
import h5py

# For dense CRF functions
from .energy import IntrinsicEnergy
from .params import IntrinsicParameters
from .krahenbuhl2013.krahenbuhl2013 import DenseCRF
from scipy.sparse import coo_matrix



# Timelapse data without GT
class TLData(object):
    def __init__(self, data_loader, flip):
        self.data_loader = data_loader
        # self.fineSize = fineSize
        # self.max_dataset_size = max_dataset_size
        self.flip = flip
        # st()
        self.npixels = (256 * 256* 29)

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def compute_N(self, RS, RB_list, num_R_features):
        N = np.ones(RS.shape[1],dtype=np.float32)

        for j in range(0,25):
            old_N = np.copy(N)
            v = RS.dot(N)
            av1 = np.copy(v)
            av2 = np.copy(v)

            for k in range(0,num_R_features+1):
                av1 = RB_list[k].dot(av1)
                av2 = RB_list[num_R_features-k].dot(av2)
            av = av2 + av1
            av = RS.transpose().dot(av) 

            N = np.sqrt(np.divide(N,av))
            error = np.max(  np.abs(  np.divide( (N  - old_N) , (old_N + 1e-8)) ) )
            if error < 1e-6:
               break

        return N

    def sparse_loader_batch(self, images, num_features):
        npixels = images.size(0) * images.size(2) * images.size(3)

        densecrf = DenseCRF(npixels, 256)
        np_imgs = images.numpy()
        np_imgs = np.transpose(np_imgs, (0,2,3,1))

        params = IntrinsicParameters()        
        energy = IntrinsicEnergy(np_imgs, params)
        Sparse_Mat_arr = densecrf.construct_sparse(energy.get_R_features_batch().copy())

        offset = np.asarray(Sparse_Mat_arr[0]) + 1
        barycentric = np.asarray(Sparse_Mat_arr[1]) # N(d+1)
        n1 = np.asarray(Sparse_Mat_arr[2]) +1
        n2 = np.asarray(Sparse_Mat_arr[3]) +1

        mn = np.asarray(Sparse_Mat_arr[4])
        N =  np_imgs.shape[0] * np_imgs.shape[1] * np_imgs.shape[2]
        valid_end = N * (num_features + 1)
        offset = offset[0:valid_end]
        barycentric = barycentric[0:valid_end]

        S_i_idx = np.arange(N)
        S_i_idx = np.repeat(S_i_idx, (num_features+1))

        m = int(mn[0])
        n = int(mn[1])
        S_mat = coo_matrix((barycentric,(offset, S_i_idx)), shape =(m+2,n), dtype=np.float32)

        S_mat = S_mat.tocsr()

        S_i = torch.from_numpy(np.column_stack( (offset, S_i_idx)  )).t_().long()
        S_v = torch.from_numpy(barycentric).float()
        S_pytorch = torch.sparse.FloatTensor(S_i, S_v, torch.Size([m+2,n]))

        B_arr = []
        B_list_pytorch = []
        for i in range(0,num_features+1):
            n1_i = n1[i*m:(i+1)*m]
            n2_i = n2[i*m:(i+1)*m]

            B_i_idx = np.arange(m) + 1

            half_w = np.repeat(0.5, m)
            one_w = np.repeat(1, m)
            # row index 
            row_index = np.int_( np.concatenate((B_i_idx,B_i_idx, B_i_idx)) )
            # column idnex
            col_index = np.int_( np.concatenate( ( B_i_idx, n1_i, n2_i))  )
            copy_col_index = col_index
            #  weight index
            w_index = np.concatenate( ( one_w, half_w, half_w))

            row_index = row_index[copy_col_index > 0]
            col_index = col_index[copy_col_index > 0]
            w_index = w_index[copy_col_index > 0]

            B_i = torch.from_numpy(np.column_stack( (row_index, col_index) )).t_().long()
            B_v = torch.from_numpy(w_index).float()
            B_mat_pytorch = torch.sparse.FloatTensor(B_i, B_v, torch.Size([m+2,m+2]))
            B_list_pytorch.append(B_mat_pytorch)

            B_mat = coo_matrix((w_index,(row_index, col_index)), shape =(m+2,m+2), dtype=np.float32)

            B_arr.append(B_mat)

        #  construct sparse matrix
        # construct normalization matrix
        N_vec = self.compute_N(S_mat, B_arr, num_features)
        N_vec = torch.from_numpy(N_vec).float().view(-1,1)

        return S_pytorch, B_list_pytorch, N_vec

    def __next__(self):
        self.iter += 1
        scale = 4
        
        final_img, target_1 ,sparse_path_1r = next(self.data_loader_iter)
        final_img = final_img.squeeze(0)
        target_1['rgb_img'] = target_1['rgb_img'].squeeze(0)

        for i in range(0,scale):
            target_1['shading_w_'+str(i)] =  target_1['shading_w_'+str(i)].squeeze(0)
            target_1['mask_'+str(i)] =  target_1['mask_'+str(i)].squeeze(0)
            target_1['r_w_s'+str(i)] =  target_1['r_w_s'+str(i)].squeeze(0)

        print("CONSTRUCT SPARSE MATRIX")
        RS_1, RB_list_1, RN_1  = self.sparse_loader_batch(target_1['rgb_img'], 5)
        target_1['RS'] = RS_1
        target_1['RB_list'] = RB_list_1
        target_1['RN'] = RN_1


        return {'img_1': final_img, 'target_1': target_1}



class AlignedDataLoader(BaseDataLoader):
    def __init__(self,_root, _list_dir):
        # BaseDataLoader.initialize(self)
        # self.fineSize = opt.fineSize

        # transformations = [
            # TODO: Scale
            #transforms.CenterCrop((600,800)),
            # transforms.Scale(256, Image.BICUBIC),
            # transforms.ToTensor() ]
        transform = None
        # transform = transforms.Compose(transformations)

        # Dataset A
        # dataset = ImageFolder(root='/phoenix/S6/zl548/AMOS/test/', \
                # list_dir = '/phoenix/S6/zl548/AMOS/test/list/',transform=transform)
        # testset 
        dataset = ImageFolder(root=_root, \
                list_dir =_list_dir,transform=transform)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size= 1, shuffle= True, num_workers=int(4))

        self.dataset = dataset
        flip = False
        self.paired_data = TLData(data_loader, flip)

    def name(self):
        return 'AlignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return len(self.dataset)


class AlignedDataLoader_TEST(BaseDataLoader):
    def __init__(self,_root, _list_dir):
        # BaseDataLoader.initialize(self)
        # self.fineSize = opt.fineSize

        # transformations = [
            # TODO: Scale
            #transforms.CenterCrop((600,800)),
            # transforms.Scale(256, Image.BICUBIC),
            # transforms.ToTensor() ]
        transform = None
        # transform = transforms.Compose(transformations)

        dataset = ImageFolder_TEST(root=_root, \
                list_dir =_list_dir,transform=transform)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size= 1, shuffle= False, num_workers=int(1))

        self.dataset = dataset
        flip = False
        self.IIW_test_data = IIWData_TEST(data_loader, flip)

    def name(self):
        return 'IIW_TEST_DATA_LOADER'

    def load_data(self):
        return self.IIW_test_data

    def __len__(self):
        return len(self.dataset)


class IIWTestData(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        final_img, target_1  = next(self.data_loader_iter)
        return {'img_1': final_img, 'target_1': target_1}


class IIWTESTDataLoader(BaseDataLoader):
    def __init__(self,_root, _list_dir, mode):

        transform = None
        # transform = transforms.Compose(transformations)

        # Dataset A
        # dataset = ImageFolder(root='/phoenix/S6/zl548/AMOS/test/', \
                # list_dir = '/phoenix/S6/zl548/AMOS/test/list/',transform=transform)
        # testset 
        dataset = IIW_ImageFolder(root=_root, \
                list_dir =_list_dir, mode= mode, is_flip = False, transform=transform)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size= 16, shuffle= False, num_workers=int(1))
        self.dataset = dataset
        self.iiw_data = IIWTestData(data_loader)

    def name(self):
        return 'IIWTESTDataLoader'

    def load_data(self):
        return self.iiw_data

    def __len__(self):
        return len(self.dataset)

