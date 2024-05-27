import torch
from ei.ei import EI
from physics.inpainting import Inpainting

from transforms.shift import Shift
from transforms.rotate import Rotate
device='cuda:0'
epochs = 1002
ckp_interval = 200
schedule = [800, 900,1000]

alpha = {'ei': 1} # equivariance strength

lr = {'G': 1e-2, 'WD': 1e-8}

#define inverse problem (forward model), e.g. inpainting task
physics = Inpainting(img_heigth=144, img_width=144, device=device)

# define transformation group {T_g}, e.g. random shift
n_trans= 7
transform = Shift(n_trans=n_trans)   # define transformation



# define Equivariant Imaging model
ei = EI(in_channels=128, out_channels=128, img_width=144, img_height=144,
        dtype=torch.float, device=device)


# train normal EI
ei.train_ei(physics, transform, epochs, lr, alpha, ckp_interval,
            schedule, residual=True, pretrained=None, task='inpainting',
            loss_type='l2', cat=True, lr_cos=False, report_psnr=True)
