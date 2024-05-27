import os
import torch
from torch.optim import Adam
######### Import DIP networks #################
from utils.inpainting_utils import *
from .closure.ei import closure_ei
from .closure.ei_adv import closure_ei_adv
from .closure.mc import closure_mc
from .closure.supervised import closure_sup
from .closure.supervised_ei import closure_sup_ei
from utils.nn import adjust_learning_rate
from utils.logger import get_timestamp, LOG

from models.Unet_Sert import Unet_Sert

class EI(object):
    def __init__(self, in_channels, out_channels, img_width, img_height, dtype, device):
        super(EI, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_width = img_width
        self.img_height = img_height

        self.dtype = dtype
        self.device = device

    def train_ei(self,  physics, transform, epochs,
                      lr, alpha, ckp_interval, schedule, residual=True,
                      pretrained=None, task='',
                      loss_type='l2', cat=True, report_psnr=False,
                      lr_cos=False):
        save_path = './ckp/'

        os.makedirs(save_path, exist_ok=True)


        pad = 'reflection' #'zero'        
        bands = 128

        generator = Unet_Sert(img_ch=bands, output_ch=bands, window_size = 16,num_heads = 8, memory_blocks = 256, down_rank = 4).to(self.device)

        s  = sum(np.prod(list(p.size())) for p in generator.parameters())
        print ('Number of params: %d' % s)
        #generator = my_skip(128,128,128).to(self.device)

        if pretrained:
            checkpoint = torch.load(pretrained, map_location=self.device)
            generator.load_state_dict(checkpoint['state_dict'])

        if loss_type=='l2':
            criterion_mc = torch.nn.MSELoss().to(self.device)
            criterion_ei = torch.nn.MSELoss().to(self.device)
        if loss_type=='l1':
            criterion_mc = torch.nn.L1Loss().to(self.device)
            criterion_ei = torch.nn.L1Loss().to(self.device)

        optimizer = Adam(generator.parameters(), lr=lr['G'], weight_decay=lr['WD'])

        if report_psnr:
            log = LOG(save_path, filename='training_loss',
                      field_name=['epoch', 'loss_mc', 'loss_ei', 'loss_total', 'psnr', 'mse'])
        else:
            log = LOG(save_path, filename='training_loss',
                      field_name=['epoch', 'loss_mc', 'loss_ei', 'loss_total'])

        for epoch in range(epochs):
            adjust_learning_rate(optimizer, epoch, lr['G'], lr_cos, epochs, schedule)

            loss = closure_ei(generator, physics, transform,
                    optimizer, criterion_mc, criterion_ei,
                    alpha, self.dtype, self.device, report_psnr)


            log.record(epoch + 1, *loss)
            
            
            if report_psnr:
                print('{}\tEpoch[{}/{}]\tmc={:.4e}\tei={:.4e}\tloss={:.4e}\tpsnr={:.4f}\tmse={:.4e}'.format(get_timestamp(), epoch, epochs, *loss))
            else:
                print(
                    '{}\tEpoch[{}/{}]\tmc={:.4e}\tei={:.4e}\tloss={:.4e}'.format(get_timestamp(), epoch, epochs, *loss))

            if epoch % ckp_interval == 0 or epoch + 1 == epochs:
                state = {'epoch': epoch,
                         'state_dict': generator.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state, os.path.join(save_path, 'ckp_Hyper_EI.pth.tar'.format(epoch)))
                
                
        log.close()


















































