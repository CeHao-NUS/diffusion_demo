
# diffusion policy import
from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

# env import
import gym
from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from skvideo.io import vwrite
from IPython.display import Video
import gdown
import os

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


# class Conv1dBlock(nn.Module):
#     '''
#         Conv1d --> GroupNorm --> Mish
#     '''

#     def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
#         super().__init__()

#         self.block = nn.Sequential(
#             nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
#             nn.GroupNorm(n_groups, out_channels),
#             nn.Mish(),
#         )

#     def forward(self, x):
#         return self.block(x)
    
class Conv1dBlock(nn.Module):
    '''
        Conv1d --> Dynamic GroupNorm --> Mish
    '''
    def __init__(self, inp_channels, out_channels, kernel_size):
        super().__init__()

        # Dynamically adjust n_groups to be a divisor of out_channels
        n_groups = math.gcd(out_channels, 8)  # For example, ensuring it's a divisor of 8 or adjust as needed

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)



class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # Adjust for channel mismatch
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.blocks[0](x)
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class Unet1D(nn.Module):
    def __init__(self, input_dim, diffusion_step_embed_dim=256, down_dims=[256, 512, 1024], kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        self.mid_modules = nn.ModuleList([
            ResidualBlock1D(mid_dim, mid_dim, kernel_size=kernel_size, n_groups=n_groups),
            ResidualBlock1D(mid_dim, mid_dim, kernel_size=kernel_size, n_groups=n_groups),
        ])

        self.down_modules = nn.ModuleList([
            nn.ModuleList([
                ResidualBlock1D(dim_in, dim_out, kernel_size=kernel_size, n_groups=n_groups),
                ResidualBlock1D(dim_out, dim_out, kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if ind < (len(in_out) - 1) else nn.Identity()
            ]) for ind, (dim_in, dim_out) in enumerate(in_out)
        ])

        self.up_modules = nn.ModuleList([
            nn.ModuleList([
                ResidualBlock1D(dim_out * 2, dim_in, kernel_size=kernel_size, n_groups=n_groups),
                ResidualBlock1D(dim_in, dim_in, kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if ind < (len(in_out) - 2) else nn.Identity()
            ]) for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:]))
        ])

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder

    def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]):
        sample = sample.moveaxis(-1, -2)  # Adjust shape to (B,C,T)
        timesteps = timestep if torch.is_tensor(timestep) else torch.tensor([timestep], dtype=torch.long, device=sample.device)
        timesteps = timesteps.expand(sample.shape[0])  # Make sure timesteps are in batch dimension
        global_feature = self.diffusion_step_encoder(timesteps)

        x = sample
        h = []

        for (resnet, resnet2, downsample) in self.down_modules:
            x = resnet(x)
            x = resnet2(x)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x)

        for (resnet, resnet2, upsample) in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x)
            x = resnet2(x)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.moveaxis(-2, -1)  # Restore shape to (B,T,C)
        return x
