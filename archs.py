import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *
__all__ = ['FMLP']
import cv2
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
import pdb
from torchvision.transforms import ToPILImage, Normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import math
from torchvision.ops import DeformConv2d

class EnhancedFrequencySeparator(nn.Module):
    def __init__(self, low_ratio, high_ratio, learnable_transition=True):
        super().__init__()
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio
        self.learnable_transition = learnable_transition
        
        if learnable_transition:
            self.alpha = nn.Parameter(torch.tensor(0.5))  
            self.beta = nn.Parameter(torch.tensor(0.5))
        
        self.register_buffer('noise_thresh', torch.tensor(0.1))

    def get_masks(self, H, W, device):
        h = torch.linspace(-0.5, 0.5, H, device=device)
        w = torch.linspace(-0.5, 0.5, W, device=device)
        hw_grid = torch.meshgrid(h, w, indexing='ij')
        radius = torch.sqrt(hw_grid[0]**2 + hw_grid[1]**2)
        
        low_mask = (radius < self.low_ratio).float()
        mid_mask = ((radius >= self.low_ratio) & (radius < self.high_ratio)).float()
        high_mask = (radius >= self.high_ratio).float()
        
        if self.learnable_transition:
            low_mask += self.alpha * mid_mask
            high_mask += self.beta * mid_mask
        
        return low_mask, high_mask

    def forward(self, x):
        B, C, H, W = x.shape
        
        x_mean = torch.mean(x, dim=(2,3), keepdim=True)
        x_std = torch.std(x, dim=(2,3), keepdim=True)
        x_norm = (x - x_mean) / (x_std + 1e-6)
        
        x_fft = torch.fft.fft2(x_norm)
        x_fft = torch.fft.fftshift(x_fft)
        
        low_mask, high_mask = self.get_masks(H, W, x.device)
        low_mask = low_mask.view(1, 1, H, W)
        high_mask = high_mask.view(1, 1, H, W)
        
        low_fft = x_fft * low_mask
        high_fft = x_fft * high_mask
        
        high_fft_abs = torch.abs(high_fft)
        high_fft = torch.where(high_fft_abs < self.noise_thresh, 
                              torch.zeros_like(high_fft), high_fft)
        
        low_freq = torch.fft.ifft2(torch.fft.ifftshift(low_fft)).real
        high_freq = torch.fft.ifft2(torch.fft.ifftshift(high_fft)).real

        low_freq = low_freq * x_std + x_mean
        high_freq = high_freq * x_std + x_mean
        
        return low_freq, high_freq
        
        
class AdaptiveDirectionShift(nn.Module):
    def __init__(self, shift_size=5, max_shift=2):
        super().__init__()
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.max_shift = max_shift
        
        self.dir_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, 3, padding=1)  
        )

    @staticmethod
    def safe_roll(x, shift, dim):
        if shift == 0:
            return x
        return torch.roll(x, shift, dim)

    def _shift_op(self, x, shifts, dim):
        xs = torch.chunk(x, self.shift_size, 1)
        shift_list = shifts.tolist() if isinstance(shifts, torch.Tensor) else shifts
        shifted_xs = [self.safe_roll(x_c, s, dim) for x_c, s in zip(xs, shift_list)]
     
        
        return torch.cat(shifted_xs, dim=1)

    def forward(self, x, edge_guidance=None):
        B, C, H, W = x.shape
        xn = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
    
        if edge_guidance is not None:
            edge_map = torch.mean(edge_guidance, dim=1, keepdim=True)
            dir_weights = torch.softmax(self.dir_net(edge_map), dim=1)
            shifts_h = [torch.randint(-self.max_shift, self.max_shift+1, (1,)).item()
                      for _ in range(self.shift_size)]
            shifts_w = [torch.randint(-self.max_shift, self.max_shift+1, (1,)).item()
                      for _ in range(self.shift_size)]
    
            shifted_h = self._shift_op(xn, shifts_h, 2)
            shifted_w = self._shift_op(xn, shifts_w, 3)
    
            shifted_h = torch.narrow(shifted_h, 2, self.pad, H)
            shifted_h = torch.narrow(shifted_h, 3, self.pad, W)
    
            shifted_w = torch.narrow(shifted_w, 2, self.pad, H)
            shifted_w = torch.narrow(shifted_w, 3, self.pad, W)
            
        
            x_shift = dir_weights[:,0:1] * shifted_h + dir_weights[:,1:2] * shifted_w
    
        else:
            
            shifts = [torch.randint(-self.max_shift, self.max_shift+1, (1,)).item()
                    for _ in range(self.shift_size)]
            x_shift = self._shift_op(xn, shifts, 2)
            x_shift = self._shift_op(xn, shifts, 3)
    
       
            x_shift = torch.narrow(x_shift, 2, self.pad, H)
            x_shift = torch.narrow(x_shift, 3, self.pad, W)
    
        return x_shift
    
    
    
class FrequencyFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.attn = nn.Sequential(
            nn.Linear(2*channels, channels//4),
            nn.ReLU(),
            nn.Linear(channels//4, 3),  
            nn.Softmax(dim=-1)
        )
        
    def forward(self, low, high):
        B, N, C = low.shape
        feat_cat = torch.cat([low, high], dim=-1)
        weights = self.attn(feat_cat)  # [B,N,3]
        
        cross_feat = low * high 
        return (weights[:,:,0:1] * low + 
                weights[:,:,1:2] * high + 
                weights[:,:,2:3] * cross_feat)

class shiftmlp(nn.Module):
 
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., shift_size=5, 
                 freq_low=0.2, freq_high=0.6):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
     
        self.freq_sep = EnhancedFrequencySeparator(freq_low, freq_high)
        
        
        self.low_shift = AdaptiveDirectionShift(shift_size, max_shift=2)
        self.high_shift = AdaptiveDirectionShift(shift_size, max_shift=1)
        
        
        self.fusion = FrequencyFusion(in_features)
        
        
        self.gate = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Sigmoid()
        )
        
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self._init_weights()
        
        self.visualizer = HeatmapVisualizer()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        
       
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        
   
        low_freq, high_freq = self.freq_sep(xn)
        
   
        x_low = self.low_shift(low_freq)
        
        x_low = x_low.reshape(B, -1, H*W).transpose(1, 2)
        
      
        x_high = self.high_shift(high_freq, edge_guidance=high_freq)
        
        x_high = x_high.reshape(B, -1, H*W).transpose(1, 2)
        
      
        fused = self.fusion(x_low, x_high)#x_low+x_high#
        
      
        g = self.gate(x)
        x = x * g + fused * (1 - g)
        
      
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        
        
        xn = x.transpose(1, 2).view(B, -1, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shifts=(shift, shift), dims=(2, 3)) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, -1, H*W).transpose(1, 2)
        
        x = self.fc2(x_s)
        x = self.drop(x)
        
        return x

class DWConv(nn.Module):
   
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

        
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class FMLP(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP
    
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 32, 64, 128],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(3, 8, 3, stride=1, padding=1)  
        self.encoder2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)  
        self.encoder3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(8)
        self.ebn2 = nn.BatchNorm2d(16)
        self.ebn3 = nn.BatchNorm2d(32)
        
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(64)
        self.dnorm4 = norm_layer(32)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(128, 64, 3, stride=1,padding=1)  
        self.decoder2 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)  
        self.decoder3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1) 
        self.decoder4 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(8, 8, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(64)
        self.dbn2 = nn.BatchNorm2d(32)
        self.dbn3 = nn.BatchNorm2d(16)
        self.dbn4 = nn.BatchNorm2d(8)
        
        self.final = nn.Conv2d(8, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)
        self.visualizer = HeatmapVisualizer()

    def forward(self, x):
    
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,t4)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
        
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        
        
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
    
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))


        return self.final(out)

