
#@markdown ### **Imports**
# diffusion policy import
import numpy as np
import torch
import torch.nn as nn
import os

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

from train import create_networks

def eval_main(num_diffusion_iters, batch_size=1024):
    # load model
    noise_pred_net, noise_scheduler =  create_networks(num_diffusion_iters)

    # device transfer
    device = torch.device('cuda')
    _ = noise_pred_net.to(device)

    ema_noise_pred_net = noise_pred_net
    ema_noise_pred_net.load_state_dict(torch.load('ema_noise_pred_net.pth'))

    # eval
    patch_values = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # patch_values = [-0.7]
    losses = {'mse':[], 'std':[]}
    for const_path in patch_values:
        mse_loss, std_loss = eval(ema_noise_pred_net, noise_scheduler, num_diffusion_iters=num_diffusion_iters, device='cuda', 
                    pred_horizon=16, action_dim=1, batch_size=batch_size,
                    const_patch=const_path)
        losses['mse'].append(np.sqrt(mse_loss))
        losses['std'].append(std_loss)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    plt.plot(patch_values, losses['mse'], label='mse', color='b')
    plt.plot(patch_values, losses['std'], label='std', color='r')
    plt.title('losses')
    plt.legend()
    plt.xlabel('patch_values')
    plt.ylabel('losses')
    plt.savefig('./{}/losses.png'.format(FILE_NAME))
    plt.show()

    np_file_name = './{}/losses.npy'.format(FILE_NAME)
    np.save(np_file_name, losses)
                


def eval(ema_noise_pred_net, noise_scheduler, num_diffusion_iters, device, pred_horizon, action_dim, batch_size=1, const_patch=0):

    # 1. mask index, size
    # 2. patch value

    '''
    Solutions:
    1. mask size up, more diff steps.
    

    Metrics:
    1. x0 OOD, (variance)
    2. Dist(x0, const_patch)
    3. Dist(x0, demo dist)
    '''

    # const_patch = 0.7
    print("==== const_patch ====", const_patch)
    patch_value = const_patch
    mask_index = np.arange(3, 7)

    mask = torch.zeros((batch_size, pred_horizon, action_dim)).to(device)
    mask[:, mask_index, :] = 1


    

    # inference actions
    # batch_size = 1024
    obss = torch.zeros((batch_size, 0, 0)).to(device)
    obs_cond = obss.flatten(start_dim=1)

    store = []

    with torch.no_grad():
        naction = torch.randn(
            (batch_size, pred_horizon, action_dim), device=device)
        
        # init scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)

        for k in noise_scheduler.timesteps:
            # predict noise
            noise_pred = ema_noise_pred_net(
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample


            store.append(naction.detach().to('cpu').numpy().squeeze())

            # ========== apply mask ==========
            # 1. noise of mask shape
            forward_noise = torch.randn_like(naction) 
            patch_const = torch.ones_like(naction) * patch_value

            noisy_patch = noise_scheduler.add_noise(
                        patch_const, forward_noise, k)

            # 2. add noise from const_patch
            # initial state
            # naction = naction * (1 - mask) + noisy_patch * mask
            # naction = noisy_patch

            # ========== direct MSE optimization ==========
            # '''
            naction_ori = naction[:, :, 0].cpu().numpy()
            # naction_ori = np.random.randn(batch_size, pred_horizon)
            noisy_patch_ori = noisy_patch[:, :, 0].cpu().numpy()
            mask_ori = mask[:, :, 0].cpu().numpy()



            fit_variable = cp.Variable((batch_size, pred_horizon))

            masked_patch_diff = cp.multiply(mask_ori , (fit_variable - noisy_patch_ori))
            mse_loss_mask = cp.sum_squares(masked_patch_diff)

            backward_diff = fit_variable - naction_ori
            mse_loss_backward = cp.sum_squares(backward_diff)

            total_loss = mse_loss_mask +  32 * mse_loss_backward
            # total_loss = mse_loss_backward

            problem = cp.Problem(cp.Minimize(total_loss))
            problem.solve()
            fit_results = fit_variable.value
            # print("fit_variable:", fit_results)

            fit_results = fit_results.reshape(batch_size, pred_horizon, 1)
            naction = torch.tensor(fit_results, dtype=torch.float32).to(device)

            # a = 1
            # '''


            '''
            # naction_ori = torch.clone(naction).detach()
            # naction_ori.requires_grad = False
            naction_ori = torch.ones_like(naction)
            
            naction_optimize = torch.randn_like(naction_ori, requires_grad=True)

            optimizer = optim.SGD([naction_optimize], lr=0.01)

            for i in range(500):
                

                mse_loss_mask = nn.functional.mse_loss(naction_optimize, naction_ori)

                mse_loss_backward = nn.functional.mse_loss(naction_optimize, naction_ori)
                # total_loss = mse_loss_mask + mse_loss_backward
                total_loss = mse_loss_mask

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # print('mse_loss_mask', mse_loss_mask)
                # print('mse_loss_backward', mse_loss_backward)
                # print('total_loss', total_loss)

            naction = naction_optimize
            a = 1
            '''


            # ========== method 2, by optimization ==========
            '''
            # 1. sample forward by adding noise, 

            zero_noise = torch.tensor(0.0).to(device)
            ones_noise = torch.tensor(1.0).to(device)
            mask_shape = torch.tensor(mask_index).to(device) 
            mask_len = len(mask_index)

            w_mean = noise_scheduler.add_noise(ones_noise, zero_noise, k)
            w_var = noise_scheduler.add_noise(zero_noise, ones_noise, k) 

            mask_shape_patch = torch.ones_like(mask_shape).repeat(batch_size, 1) * patch_value
            mean_noise = mask_shape_patch * w_mean
            var_noise = torch.ones((mask_len, mask_len)).repeat(batch_size, 1, 1).to(device) * w_var

            # add small noise to var_noise
            var_noise  = var_noise +  torch.eye(mask_len).repeat(batch_size, 1, 1).to(device) * 1e-5

            dist = MultivariateNormal(mean_noise, var_noise)
            masked_action = naction[:, mask_index, 0]

            nll_lose_all = -dist.log_prob(masked_action)
            nll_lose = torch.mean(nll_lose_all)
            
            '''


            # 2. sample backward dist
            # a = 1



    # ============ vis ============
    data = np.array(store).transpose(1, 0, 2) # B, diff steps, length of action
    file_name_np = './{}/data_{}.npy'.format(FILE_NAME, str(const_patch))
    # create sub dir for FILE_NAME
    if not os.path.exists(FILE_NAME):
        os.makedirs(FILE_NAME)
    np.save(file_name_np, data)

    # print('data', data.shape)


    last_action = data[:, -1, :]
    # print('last_action', last_action.shape)
    mean = np.mean(last_action, axis=1)
    std = np.std(last_action, axis=1)

    x_batch_size = range(len(last_action))


    
    # from vis import vis_1d
    plt.figure(figsize=(20,10))

    plt.subplot(221)
    plt.plot(x_batch_size, mean, label='mean')
    plt.fill_between(x_batch_size, mean-std, mean+std, alpha=0.3)
    plt.plot(x_batch_size, np.ones(len(last_action))*patch_value, label='patch_value', color='r')
    plt.legend()
    plt.title('mean+-std over batch size')


    plt.subplot(222)
    plt.plot(x_batch_size, std)
    plt.title('std')


    # mes loss
    mse_loss = np.mean(np.square(mean - patch_value))
    print('mse_loss to const patch', mse_loss)

    # std loss
    std_loss = np.mean(std)
    print('std_loss to demo dist', std_loss)

    plt.subplot(223)
    mean = np.mean(last_action, axis=0)
    std = np.std(last_action, axis=0)
    plt.plot(mean, label='mean')
    plt.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.3)
    plt.plot(mask_index, np.ones(len(mask_index))*patch_value, label='patch_value', color='r')
    plt.legend()
    plt.title('mean+-std over action space')

    plt.subplot(224)
    # one_data = data[:,:,0] #B, diff steps
    one_data = np.mean(data, axis=-1)

    # print('one_data', one_data.shape)
    mean = np.mean(one_data, axis=0)
    std = np.std(one_data, axis=0)
    plt.plot(mean, label='mean')
    plt.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.3)
    plt.plot(np.ones(len(mean))*patch_value, label='patch_value', color='r')
    plt.legend()
    plt.title('mean+-std over diffusion')

    # create sub dir for FILE_NAME
    if not os.path.exists(FILE_NAME):
        os.makedirs(FILE_NAME)

    file_name = './{}/result_{}.png'.format(FILE_NAME, str(const_patch))
    plt.savefig(file_name)
    plt.close()

    # vis_1d(last_action) # over action space

    # vis_1d(one_data) # over diff steps
    

    return mse_loss, std_loss