import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange
import  numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
    
class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16,memory_blocks=128):
        super(ChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.subnet = nn.Sequential(
            
            nn.Linear(num_feat, num_feat // squeeze_factor),
            #nn.ReLU(inplace=True)
           )
        self.upnet= nn.Sequential(
            nn.Linear(num_feat // squeeze_factor, num_feat),
            #nn.Linear(num_feat, num_feat),
            nn.Sigmoid())
        self.mb =  torch.nn.Parameter(torch.randn(num_feat // squeeze_factor, memory_blocks))
        self.low_dim = num_feat // squeeze_factor

    def forward(self, x):
       
        b,n,c = x.shape
        t = x.transpose(1,2)
        y = self.pool(t).squeeze(-1)

        low_rank_f = self.subnet(y).unsqueeze(2)

        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1)
        f1 = (low_rank_f.transpose(1,2) ) @mbg  
        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1) # get the similarity information
        y1 = f_dic_c@mbg.transpose(1,2) 
        y2 = self.upnet(y1)
        out = x*y2
        return out

class CAB(nn.Module):      
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30,memory_blocks=128):         
        super(CAB, self).__init__()  
        self.num_feat = num_feat     
        self.cab = nn.Sequential(             
        nn.Linear(num_feat,num_feat // compress_ratio),             
        nn.GELU(),             
        nn.Linear(num_feat // compress_ratio, num_feat),             ChannelAttention(num_feat, squeeze_factor, memory_blocks) )      
    
    def forward(self, x): 
     
        return self.cab(x)




class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=0, qk_scale=None, memory_blocks=128,down_rank=16, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
       
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)



        self.c_attns = CAB(dim,compress_ratio=4,squeeze_factor=down_rank,memory_blocks=memory_blocks)  #
        #self.c_attns_15 = CAB(dim,compress_ratio=4,squeeze_factor=15)  

    def forward(self, x, mask=None):
        x3 = self.c_attns(x)

        x = self.proj(x3)
        x = self.proj_drop(x)
        return x






class SE_block(nn.Module):
    r"""  SE Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,drop_path=0.0,memory_blocks=128,down_rank=16,
                qkv_bias=True, qk_scale=None, drop=0.,shift_size=0, attn_drop=0.,  act_layer=nn.GELU):
        super(SE_block,self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size       
        self.norm1 =  nn.LayerNorm(dim) 
        self.norm2 = nn.LayerNorm(dim) 
        self.shift_size = shift_size
        self.attns = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,memory_blocks=memory_blocks,down_rank=down_rank,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.num_heads = num_heads


    def forward(self, x):
        #print('transformer block input size:',x.shape)
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1, 2)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)  

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  
        
        attn_windows = self.attns(x_windows)
        
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        x = x.transpose(1, 2).view(B, C, H, W)
        
        return x

















def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)


    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)




class conv_block_down(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block_down,self).__init__()
        need_bias=True
        pad= 'reflection'
        self.conv = nn.Sequential(
            #nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            conv(ch_in, ch_out, 3, 2, bias=need_bias, pad=pad, downsample_mode='stride'),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
            conv(ch_out, ch_out, 3, 1, bias=need_bias, pad= pad),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class conv_block_up(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block_up,self).__init__()
        need_bias=True
        pad= 'reflection'
        self.conv = nn.Sequential(
            nn.BatchNorm2d(ch_in),
            #nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            conv(ch_in, ch_out, 3, 1, bias=need_bias, pad=pad),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
            conv(ch_out, ch_out, 3, 1,bias=need_bias, pad=pad),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self,x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        upsample_mode = 'nearest'
        pad = 'reflection'
        need_bias=True
        self.up = nn.Sequential( 
            nn.Upsample(scale_factor=2, mode=upsample_mode),            
            #nn.Conv2d(ch_in,ch_out,kernel_size=5,stride=1,padding=0,bias=True),
            conv(ch_in, ch_out, 3, 1, bias=need_bias, pad=pad),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
            conv(ch_in, ch_out, 3, 1, bias=need_bias, pad=pad),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True)           
            
        )

    def forward(self,x):
        x = self.up(x)
        return x



class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        need_bias=True
        pad= 'reflection'
        self.conv = nn.Sequential(
            conv(ch_in, ch_out, 3, 1, bias=need_bias, pad=pad),
           nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x





       

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        pad = 'reflection'
        need_bias=True
        self.W_g = nn.Sequential(
            conv(F_g, F_int, 1, 1,bias=need_bias, pad=pad),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            conv(F_l, F_int, 1,1,bias=need_bias, pad=pad),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            conv(F_int, 1, 1,1,bias=need_bias, pad=pad),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi



class Unet_Sert(nn.Module):
    def __init__(self,img_ch=128,output_ch=128,window_size = 16,num_heads = 8, memory_blocks = 256, down_rank = 8,qkv_bias=True, qk_scale=None,
        bias=False, attn_drop = 0,proj_drop=0):
        super(Unet_Sert,self).__init__()
        self.SE_1 = SE_block(dim= 128,input_resolution=window_size, num_heads=num_heads, memory_blocks= memory_blocks,window_size=window_size,
                                     down_rank=down_rank,qkv_bias=True,shift_size=0)
        self.SE_2 = SE_block(dim= 128,input_resolution=window_size//2, num_heads=num_heads, memory_blocks= memory_blocks,window_size=window_size//2,
                                     down_rank=down_rank,qkv_bias=True,shift_size=0)                                            
        self.SE_3 = SE_block(dim= 128,input_resolution=window_size//4, num_heads=num_heads, memory_blocks= memory_blocks,window_size=window_size//4,
                                     down_rank=down_rank,qkv_bias=True,shift_size=0)       
        self.SE_4 = SE_block(dim= 128,input_resolution=window_size//8, num_heads=num_heads, memory_blocks= memory_blocks,window_size=window_size//8,
                                     down_rank=down_rank,qkv_bias=True,shift_size=0)



        self.Conv1 = conv_block_down(ch_in=img_ch,ch_out=128)
        self.Conv2 = conv_block_down(ch_in=128,ch_out=128)
        self.Conv3 = conv_block_down(ch_in=128,ch_out=128)
        self.Conv4 = conv_block_down(ch_in=128,ch_out=128)
        
                
        self.Up4 = up_conv(ch_in=128,ch_out=128)
        self.Att4 = Attention_block(F_g=128,F_l=128,F_int=256)
        self.Up_conv4 = conv_block_up(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=256)
        self.Up_conv3 = conv_block_up(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=128)
        self.Att2 = Attention_block(F_g=128,F_l=128,F_int=256)
        self.Up_conv2 = conv_block_up(ch_in=256, ch_out=128)
        
        self.Up1 = up_conv(ch_in=128,ch_out=128)
        self.Att1 = Attention_block(F_g=128,F_l=128,F_int=256)
        self.Up_conv1 = conv_block_up(ch_in=256, ch_out=128)
        
        self.bn= nn.BatchNorm2d(256)
        
        self.need_sigmoid = nn.Sigmoid()
        
        pad = 'reflection'
        need_bias=True
        self.first = single_conv(img_ch,128)
        self.final= conv(128, output_ch, 1, bias=need_bias, pad=pad)



 
                  
    def forward(self,x):
        # encoding path
        x_single = self.first(x)
        x1 = self.Conv1(x)        
        x2 = self.Conv2(x1)            
        x3 = self.Conv3(x2)       
        x4 = self.Conv4(x3)  

        # decoding + concat path
        d4 = self.Up4(x4)  
        x3 = self.Att4(g=d4,x=self.SE_4(x3))      
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)     
        x2 = self.Att3(g=d3,x=self.SE_3(x2))
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=self.SE_2(x1))
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        
        d1 = self.Up1(d2)
        x = self.Att1(g=d1,x=self.SE_1(x_single))
        d1 = torch.cat((x,d1),dim=1)
        d1 = self.Up_conv1(d1)
        
        final = self.final(d1)

        return self.need_sigmoid(final)     
            
 
    
