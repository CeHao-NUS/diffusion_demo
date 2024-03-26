

#@markdown ### **Imports**
# diffusion policy import
import numpy as np
import pickle
import torch
import torch.nn as nn
import os

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm



from train import create_networks
from inpainting import line_patch, vanila_inpainting, MSE_opt
from vis import calculate_mse_loss, vis_last_action, vis_diff_steps

from args import get_args


def eval_main(args):

    # 0. initialize parameters
    diff_model_name = 'ema_noise_pred_net.pth'

    # 1. load model
    noise_pred_net, noise_scheduler =  create_networks(args.num_diffusion_iters)

    # device transfer
    device = torch.device('cuda')
    _ = noise_pred_net.to(device)

    ema_noise_pred_net = noise_pred_net
    ema_noise_pred_net.load_state_dict(torch.load(diff_model_name))

    # 2. create inpainting patches and mask
    # patch_values = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    patch_values = [-0.7]
    
    for const_path in patch_values:
        patch, mask = line_patch(patch_value=const_path)
        inpainting_dict = {'patch': patch, 'mask': mask}

        # 3. call eval and output losses and other results
        x_results = eval(args, ema_noise_pred_net, noise_scheduler, 
                         args.num_diffusion_iters, args.batch_size, 
                         inpainting_dict, device='cuda')

        # 4. save results
        # check folder exists
        if not os.path.exists(args.folder_name):
            os.makedirs(args.folder_name)
        # get full path
        save_file_name = './{}/results_{}.pkl'.format(args.folder_name, const_path)
        save_file = {'x_results': x_results, 'inpainting_dict': inpainting_dict}
        # save by pickle
        with open(save_file_name, 'wb') as f:
            pickle.dump(save_file, f)

        # save with name last
        save_file_name = './{}/results_last.pkl'.format(args.folder_name)
        with open(save_file_name, 'wb') as f:
            pickle.dump(save_file, f)


def eval(args, model, scheduler, num_diffusion_iters, batch_size, inpainting_dict, device='cuda'):
    # 0. initialize parameters
    obss = torch.zeros((batch_size, 0, 0)).to(device)
    obs_cond = obss.flatten(start_dim=1)

    pred_horizon = 16
    action_dim = 1

    # 1. set stores for diffusion steps

    x_store = [] # diff steps, batch_size, pred_horizon, action_dim

    # 2. apply diffusion steps
    with torch.no_grad():
        x = torch.randn(
            (batch_size, pred_horizon, action_dim), device=device)
        
        # init scheduler
        scheduler.set_timesteps(num_diffusion_iters)

        for k in scheduler.timesteps:
            # predict noise
            noise_pred = model(
                sample=x,
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            x = scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=x
            ).prev_sample

            # 4. store results, convert x to numpy
            x_store.append(x.detach().cpu().numpy())

            # 3. do inpainting with mask
            # x = vanila_inpainting(x, inpainting_dict['mask'], inpainting_dict['patch'])
            x = MSE_opt(x, inpainting_dict['mask'], inpainting_dict['patch'], args.loss_weight)

            

    # 5. save results
    x_results = np.array(x_store)
    # reshape by first two dims
    x_results = x_results.transpose(1, 0, 2, 3)

    return x_results



if __name__ == '__main__':
    args = get_args()
    eval_main(args)
    exec(open('vis.py').read())
    