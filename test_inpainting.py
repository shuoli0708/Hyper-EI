import torch
import numpy as np
import torch
import numpy as np
from utils.metric import cal_psnr, cal_mse
import math
import h5py
import scipy.io
import random
from scipy.io import loadmat
import argparse
import matplotlib.pyplot as plt
from utils.inpainting_utils import *
from physics.inpainting import Inpainting
from models.Unet_Sert import Unet_Sert
import pytorch_ssim
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


    
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



bands = 128
image_size = 144
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


#########################  load model ##########################

parser = argparse.ArgumentParser(description='Inpainting test.')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
# specifying path to trained models:
parser.add_argument('--ckp', default='./ckp/ckp_Hyper_EI.pth.tar', type=str, metavar='PATH',
                    help='path to checkpoint of Hyper-EI net')
parser.add_argument('--model-name', default='EI', type=str, help="name of the trained model (dafault: 'EI')")

args = parser.parse_args()
device = f'cuda:{args.gpu}'

forw = Inpainting(img_heigth=144, img_width=144,  device=device)
pad = 'reflection' #'zero'        
bands = 128

net = Unet_Sert(img_ch=bands ,output_ch=bands, window_size = 16,num_heads = 8, memory_blocks = 256, down_rank = 4).to(device)


def test(net, ckp, fbp, adv=False):
	checkpoint = torch.load(ckp, map_location=device)
	net.load_state_dict(checkpoint['state_dict_G' if adv else 'state_dict'])
	net.to(device).eval()
	return net(fbp)


# groundtruth
x = clean
y = target
# compute the A^+y or FBP
fbp = forw.A_dagger(y)
x_net = test(net, args.ckp, fbp)

Test_MPSNR = mpsnr(clean.view((1,bands,image_size,image_size)), x_net.to(device)) 
Test_MSSIM = pytorch_ssim.ssim(clean.view((1,bands,image_size,image_size)), x_net.to(device)) 

check_1 = clean.detach().cpu().view((bands,image_size,image_size)).numpy().transpose(1, 2,0)
check_2 = target.detach().cpu().view((bands,image_size,image_size)).numpy().transpose(1, 2,0)
check_3 = x_net.detach().cpu().view((bands,image_size,image_size)).numpy().transpose(1, 2,0)
gt_hole = (clean * mask_for_hole).detach().cpu().view((bands,image_size,image_size)).numpy().transpose(1, 2,0)
only_hole = x_net * mask_for_hole
generated_hole = only_hole.detach().cpu().view((bands,image_size,image_size)).numpy().transpose(1, 2,0)
diff_hole = gt_hole[:,:,band] - generated_hole[:,:,band]

#########################  visualize inpainting results ##########################
f, (ax1, ax2, ax3,ax4,ax5,ax6) = plt.subplots(1,6, sharey=True, figsize=(15,15))

ax1.imshow(check_1[:,:,band], cmap='gray')
ax1.title.set_text('Clean Image')
ax1.set_xlabel("Input Image MPSNR is: {:4f}".format(Initial_PSNR)+ " \n Input Image MSSIM is: {:4f}".format(Initial_MSSIM ))
ax2.imshow(check_2[:,:,band], cmap='gray')
ax2.title.set_text('Corrupted Image')
ax3.imshow(check_3[:,:,band], cmap='gray')
ax3.title.set_text('Hyper-EI Result')
ax4.imshow(gt_hole[:,:,band], cmap='gray')
ax4.title.set_text('Ground-Truth Hole Region')
ax5.imshow(generated_hole[:,:,band], cmap='gray')
ax5.title.set_text('Generated Hole Region')
ax6.imshow(diff_hole, cmap='gray')
ax6.title.set_text('difference')
ax3.set_xlabel(" \n Inpainting MPSNR is: {:4f}".format(Test_MPSNR ) + " \n Inpainting MSSIM is: {:4f}".format(Test_MSSIM ))

plt.show()




