#@markdown ### **Imports**
# diffusion policy import
import numpy as np
import torch
import torch.nn as nn

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm


from network import ConditionalUnet1D
from dataset import *

import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
import torch.optim as optim
import cvxpy as cp

def main(num_epochs=100, num_diffusion_iters=100):

    #  ===================== get data loader =====================
    dataset = get_dataloader()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        num_workers=1,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
        )
    
    # ===================== create networks =====================
    
    noise_pred_net, noise_scheduler =  create_networks(num_diffusion_iters)

    # device transfer
    device = torch.device('cuda')
    _ = noise_pred_net.to(device)


    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=noise_pred_net.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    # train
    train(num_epochs, dataloader, noise_pred_net, noise_scheduler, optimizer, lr_scheduler, ema, device)


def create_networks(num_diffusion_iters):
    # set params
    obs_horizon = 0
    pred_horizon = 16

    obs_dim = 0
    action_dim = 1

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
        )

    return noise_pred_net, noise_scheduler


def train(num_epochs, dataloader, noise_pred_net, noise_scheduler, optimizer, lr_scheduler, ema, device):

    # store and plot the loss at every epoch to tensorboard
        

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:

                    naction = nbatch.to(device)
                    B = naction.shape[0]
                    
                    obss = torch.zeros((B, 0, 0)).to(device)
                    obs_cond = obss.flatten(start_dim=1)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(noise_pred_net.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))


    # Weights of the EMA model
    # is used for inference
    ema_noise_pred_net = noise_pred_net
    ema.copy_to(ema_noise_pred_net.parameters())

    # save state dict
    torch.save(ema_noise_pred_net.state_dict(), 'ema_noise_pred_net.pth')




if __name__ == '__main__':
    # main(num_epochs=500, num_diffusion_iters=200)
    # eval_main(num_diffusion_iters=200, batch_size=256)
    pass

    
