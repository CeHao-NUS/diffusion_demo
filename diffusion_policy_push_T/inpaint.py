from main import *

import copy
import cvxpy as cp

def MSE_inequ_opt(x_ori, mask, patch, threshold):
    '''
    x_ori: (batch_size, x_length)
    '''

    # if x is tensor, detach and convert to numpy
    if isinstance(x_ori, torch.Tensor):
        x = x_ori.detach().cpu().numpy()
    else:
        x = x_ori


    batch_size = x.shape[0]
    action_length = x.shape[1]
    num_non_zero = np.sum(mask) # number of non zero elements in mask


    # flatten the second and third dimensions
    x_flat = x[0]
    mask_flat = mask[0]
    patch_flat = patch[0]

    # create optimization variable
    x_hat = cp.Variable(x_flat.shape)
    x_hat.value = x_flat

    # create optimization problem
    # loss 1: inpainting loss
    masked_diff = cp.multiply(mask_flat, (x_hat - patch_flat))
    mse_loss_mask = cp.sum_squares(masked_diff) / num_non_zero / batch_size

    # loss 2: in-dist loss
    backward_diff = x_hat - x_flat
    mse_loss_backward = cp.sum_squares(backward_diff)/ action_length / batch_size

    # total loss with weight
    total_loss = mse_loss_mask 

    # inequality constraint, mse_loss_backward < threshold
    constraint = [mse_loss_backward <= threshold]

    # create optimization problem and solve
    problem = cp.Problem(cp.Minimize(total_loss), constraints=constraint)

    problem.solve()

    # get optimized results
    x_hat = x_hat.value

    '''
    # calculate loss again
    masked_diff = np.multiply(mask_flat, (x_hat - patch_flat))
    mse_loss_mask = np.mean(masked_diff**2) / num_non_zero
    backward_diff = x_hat - x_flat
    mse_loss_backward = np.mean(backward_diff**2) / action_length

    # print(f'loss patch: {mse_loss_mask}')
    # print(f'loss in-dist: {mse_loss_backward}')
    # print('pose 1', x_hat[7:11,:])
    '''

    # reshape to original shape
    x_hat = x_hat.reshape(x.shape)

    # convert to tensor if x_ori is tensor
    if isinstance(x_ori, torch.Tensor):
        x_hat = torch.tensor(x_hat, dtype=torch.float32).to(x_ori.device)
    
    return x_hat



def inpainting_test(ema_noise_pred_net, noise_scheduler, obs_cond, noisy_action, num_diffusion_iters):
    # '''

    # =========== no inpainting ===========
    naction = copy.deepcopy(noisy_action)

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

    naction = naction.detach().to('cpu').numpy() # 1, 16, 2
    # (B, pred_horizon, action_dim)
    naction = naction[0]

    # vis_actions2(naction)
    actions0 = copy.deepcopy(naction)
    # print(actions0[7:11,:])



    # '''
    # =========== test vanilla inpainting ===========

    mask_idx = [7,8,9,10]
    mask = np.zeros((1, 16, 2))
    mask[:, mask_idx, :] = 1

    patch = np.zeros((1, 16, 2))
    patch_pose = np.array([[-0.5, 0.2], [-0.4, 0.35], [-0.25, 0.5], [-0.05, 0.6]])
    # patch_pose = np.array([[-0.3, 0.0], [-0.2, 0.1], [-0.15, 0.15], [-0.05, 0.2]])
    # patch_pose = np.array([[ 0.2142639 , -0.49882814],  [ 0.28361923 ,-0.44112352],  [ 0.3451777 , -0.35928074],  [ 0.40424222 ,-0.2608446 ]]) 


    # patch_pose[:,0] += 0.15
    # patch_pose[:, 1] -= 0.15

    # patch_pose[:,0] += 0.4
    # patch_pose[:, 1] -= 0.4

    patch[0, mask_idx, :] = patch_pose

    # to tensor and cuda
    mask_torch = torch.tensor(mask, dtype=torch.float32).to('cuda')
    patch_torch = torch.tensor(patch, dtype=torch.float32).to('cuda')

    # print(patch[0,mask_idx,:])


    naction = copy.deepcopy(noisy_action)


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

        # vanilla inpainting
        naction = mask_torch * patch_torch + (1 - mask_torch) * naction


    naction = naction.detach().to('cpu').numpy() # 1, 16, 2
    # (B, pred_horizon, action_dim)
    naction = naction[0]

    action_vanilla = copy.deepcopy(naction)


    # ================== test MSE_inequ_opt ==================
    naction = copy.deepcopy(noisy_action)


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

        # vanilla inpainting
        # naction = mask_torch * patch_torch + (1 - mask_torch) * naction

        # inpainting opt
        # to numpy first
        naction_np = naction.detach().to('cpu').numpy()
        mask_np = mask_torch.detach().to('cpu').numpy()
        patch_np = patch_torch.detach().to('cpu').numpy()
        naction_out = MSE_inequ_opt(naction_np, mask_np, patch_np, 0.001)
        naction = torch.tensor(naction_out, dtype=torch.float32).to('cuda')




    naction = naction.detach().to('cpu').numpy() # 1, 16, 2
    # (B, pred_horizon, action_dim)
    naction = naction[0]

    action_opt = copy.deepcopy(naction)

    return actions0, action_vanilla, action_opt, patch, mask_idx

# '''

if __name__ == "__main__":
    args = get_args()
    env = create_env()
    dataloader = creat_dataset(args)
    noise_pred_net, noise_scheduler = create_network(args)
    load_modal(args, noise_pred_net)

    ema_noise_pred_net = copy.deepcopy(noise_pred_net)

    # load inpainting dict
    inpaint_dict = np.load('inpainting.npy', allow_pickle=True).item()

    obs_cond = inpaint_dict['obs_cond']
    noisy_action = inpaint_dict['noisy_action']
    num_diffusion_iters = inpaint_dict['num_diffusion_iters']


    #
    actions0, action_vanilla, action_opt, patch, mask_idx = inpainting_test(ema_noise_pred_net, noise_scheduler, obs_cond, noisy_action, num_diffusion_iters)


    plt.figure(figsize=(6,6))
    plt.plot(actions0[:,0], -actions0[:,1], label='original', color='blue')
    plt.scatter(actions0[:,0], -actions0[:,1], c='blue')


    # plot vanilla
    plt.plot(action_vanilla[:,0], -action_vanilla[:,1], label='vanilla', color='green')
    plt.scatter(action_vanilla[:,0], -action_vanilla[:,1], c='green')

    # plot opt
    plt.plot(action_opt[:,0], -action_opt[:,1], label='opt', color='orange')
    plt.scatter(action_opt[:,0], -action_opt[:,1], c='orange')


    # inpainting patch
    plt.plot(patch[0,mask_idx,0], -patch[0,mask_idx,1], label='cond', color = 'red', marker='o', linewidth=2)

    # plt.colorbar()
    plt.legend()

    plt.xlim([-1,1])
    plt.ylim([-1,1])
    # plt.axis('equal')

    ax = plt.gca()  # 'gca' stands for 'get current axis'
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks

    plt.savefig('inpaint.png')
    plt.show()

    a = 1
# '''