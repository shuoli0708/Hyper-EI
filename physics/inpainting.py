import os
import torch
import h5py
import scipy.io
import random
import math
import math
from scipy.io import loadmat
import numpy as np



image_size = 144
data_dict = loadmat('Mask_Samples/mask_144_144_type_1.mat')
mask = data_dict['mask']  # 36 36 1 1
mask  = np.array(mask, order='F')
single_mask = torch.Tensor(mask).cuda()
single_mask = single_mask.reshape((1,1,image_size,image_size))



class Inpainting():
    def __init__(self, img_heigth=0, img_width=0, mode='random', resize=False, device='cuda:0'):
        self.mask = single_mask.reshape((image_size,image_size))

    def A(self, x):
		
        return torch.einsum('kl,ijkl->ijkl', self.mask, x)

    def A_dagger(self, x):
        return torch.einsum('kl,ijkl->ijkl', self.mask, x)
