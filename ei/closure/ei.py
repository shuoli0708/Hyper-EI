import torch
import numpy as np
from utils.metric import cal_psnr, cal_mse
import math
import h5py
import scipy.io
import random
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pytorch_ssim
def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

def psnr(img1, img2):
    mse = torch.mean((img1  - img2 ) ** 2)
    if mse < 1.0e-10:
        print('they are the same')
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def mpsnr(x_true, x_pred):
    n_bands = x_true.shape[1] 
    batch_size = x_true.shape[0] 
    mean_for_each = 0
    for i in range(batch_size):
        a = x_true[i,:, :, :]
        b = x_pred[i,:, :, :]
        p = [psnr( a[k,:, :], b[k,:, :]) for k in range(n_bands)]
        mean_for_each += np.mean(p)        
    return mean_for_each/batch_size


device = 'cuda'

image_size = 144
bands = 128


#########################  load test image ##########################
data_dict = h5py.File('Test_HSI_Samples/Chikusei_Test_5_images.mat')  # feel free to try different test samples
clean = data_dict['gt_blob']  # 
clean = clean[:,:,:,2]    # feel free to try different test samples by e,g. changing from clean[:,:,:,2] to clean[:,:,:,4]..
clean  = np.array(clean, order='F')
clean = torch.Tensor(clean).permute(2,0,1).view(1,bands,image_size,image_size).to(device)

#########################  load strip mask ##########################
data_dict = loadmat('Mask_Samples/mask_144_144_type_1.mat')  # feel free to try different masks stored in the folder 'Mask_Samples'
mask = data_dict['mask']  # 
mask  = np.array(mask, order='F')
single_mask = torch.Tensor(mask).to(device)
single_mask = single_mask.reshape((1,1,image_size,image_size))    # single_mask: 2D 


EI_mask = torch.zeros(1,bands,image_size,image_size).to(device)

for i in range(bands):
	EI_mask[0,i,:,:]= single_mask    # e,g. EI_mask: 3D, e,g. with size: (1,128,144,144)

# incomplete observation
target = clean * single_mask

# band to visualize
band = 90

y = torch.ones(bands,image_size,image_size).to(device)
mask_for_hole = (y - EI_mask).to(device)
mask_bkg = EI_mask.clone().to(device)

Initial_PSNR =  mpsnr(clean.view((1,bands,image_size,image_size)), target) 
Initial_MSSIM = pytorch_ssim.ssim(clean.view((1,bands,image_size,image_size)), target) 
print("Input HSI MPSNR:",Initial_PSNR)
print("Input HSI MSSIM:",Initial_MSSIM)


def closure_ei(net, physics, transform,
                    optimizer, criterion_mc, criterion_ei,
                    alpha, dtype, device, reportpsnr=False):
    loss_mc_seq, loss_ei_seq, loss_seq, psnr_seq, mse_seq = [], [], [], [], []

    x =clean # ground-truth signal x
    y0 = target  # imcomplete observation: y0
    
    x0 = physics.A_dagger(y0) # range input (A^+y)								imcomplete observation: y0

    x1 = net(x0)
    y1 = physics.A(x1)

	# equivariant imaging: x2, x3
    x2 = transform.apply(x1)
    x3 = net(physics.A_dagger(physics.A(x2)))


    loss_mc = criterion_mc(y1, y0)
    loss_ei = criterion_ei(x3, x2)

    loss = loss_mc + alpha['ei'] * loss_ei

    loss_mc_seq.append(loss_mc.item())
    loss_ei_seq.append(loss_ei.item())
    loss_seq.append(loss.item())
    
    Test_MPSNR = mpsnr(clean.view((1,bands,image_size,image_size)), x1.to(device)) 
    if reportpsnr:
        psnr_seq.append(Test_MPSNR)
        mse_seq.append(cal_mse(x1, clean.view((1,bands,image_size,image_size))))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_closure = [np.mean(loss_mc_seq), np.mean(loss_ei_seq), np.mean(loss_seq)]

    if reportpsnr:
        loss_closure.append(np.mean(psnr_seq))
        loss_closure.append(np.mean(mse_seq))

    return loss_closure
