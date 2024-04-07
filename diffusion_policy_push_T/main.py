from environment import *
from dataset import *
from network import *

from args import get_args    
import matplotlib.pyplot as plt

def create_env():
    from huggingface_hub.utils import IGNORE_GIT_FOLDER_PATTERNS
    #@markdown ### **Env Demo**
    #@markdown Standard Gym Env (0.21.0 API)

    # 0. create env object
    env = PushTEnv()

    # 1. seed env for initial state.
    # Seed 0-200 are used for the demonstration dataset.
    env.seed(1000)

    # 2. must reset before use
    obs, IGNORE_GIT_FOLDER_PATTERNS = env.reset()

    return env


def creat_dataset(args):
    # download demonstration data from Google Drive
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"
    if not os.path.isfile(dataset_path):
        id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
        gdown.download(id=id, output=dataset_path, quiet=False)

    # parameters
    # pred_horizon = 16
    # obs_horizon = 2
    # action_horizon = 8
    pred_horizon = args.pred_horizon
    obs_horizon = args.obs_horizon
    action_horizon = args.action_horizon
    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # create dataloader
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

    return dataloader

def create_network(args):
    # obs_dim = 5
    # action_dim = 2
    obs_dim = args.obs_dim
    action_dim = args.action_dim
    obs_horizon = args.obs_horizon
    pred_horizon = args.pred_horizon

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    # example inputs
    noised_action = torch.randn((1, pred_horizon, action_dim))
    obs = torch.zeros((1, obs_horizon, obs_dim))
    diffusion_iter = torch.zeros((1,))

    # the noise prediction network
    # takes noisy action, diffusion iteration and observation as input
    # predicts the noise added to action
    noise = noise_pred_net(
        sample=noised_action,
        timestep=diffusion_iter,
        global_cond=obs.flatten(start_dim=1))

    # illustration of removing noise
    # the actual noise removal is performed by NoiseScheduler
    # and is dependent on the diffusion noise schedule
    denoised_action = noised_action - noise

    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    num_diffusion_iters = 100
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

    # device transfer
    device = torch.device(args.device)
    _ = noise_pred_net.to(device)

    return noise_pred_net, noise_scheduler

def train(args, noise_pred_net, noise_scheduler, dataloader):

    # num_epochs = 100
    num_epochs = args.num_epochs
    obs_horizon = args.obs_horizon
    device = torch.device(args.device)

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

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nobs = nbatch['obs'].to(device)
                    naction = nbatch['action'].to(device)
                    B = nobs.shape[0]

                    # observation as FiLM conditioning
                    # (B, obs_horizon, obs_dim)
                    obs_cond = nobs[:,:obs_horizon,:]
                    # (B, obs_horizon * obs_dim)
                    obs_cond = obs_cond.flatten(start_dim=1)

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

    # save model
    torch.save(ema_noise_pred_net.state_dict(),  args.checkpoint_name)

def load_modal(args, noise_pred_net):
    load_pretrained = False
    if load_pretrained:
        ckpt_path = "pusht_state_100ep.ckpt"
        if not os.path.isfile(ckpt_path):
            id = "1mHDr_DEZSdiGo9yecL50BBQYzR8Fjhl_&confirm=t"
            gdown.download(id=id, output=ckpt_path, quiet=False)

        state_dict = torch.load(ckpt_path, map_location='cuda')
        ema_noise_pred_net = noise_pred_net
        ema_noise_pred_net.load_state_dict(state_dict)
        print('Pretrained weights loaded.')
    else:
        # print("Skipped pretrained weight loading.")
        # load the model from the checkpoint
        ema_noise_pred_net = noise_pred_net
        ema_noise_pred_net.load_state_dict(torch.load(args.checkpoint_name))


def eval(args, env, noise_pred_net, noise_scheduler, stats):

    max_steps = 200
    num_diffusion_iters = args.num_diffusion_iters
    obs_horizon = args.obs_horizon
    pred_horizon = args.pred_horizon
    action_dim = args.action_dim
    action_horizon = args.action_horizon


    ema_noise_pred_net = noise_pred_net
    device = torch.device(args.device)

    obs, info = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = [env.render(mode='rgb_array')]
    actions = list()
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon (2) number of observations
            obs_seq = np.stack(obs_deque)
            # normalize observation
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            # device transfer
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

            # infer action
            with torch.no_grad():
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise =============================
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                naction = noisy_action

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

                # test inpainting ==============================
                # inpainting_dict = {'obs_cond': obs_cond, 'noisy_action': noisy_action, 'num_diffusion_iters': num_diffusion_iters}
                # save them
                # np.save('inpainting.npy', inpainting_dict)
                    
                if step_idx == 0:
                    print('apply inpainting')
                    from inpaint import inpainting_test
                    actions0, action_vanilla, action_opt, patch, mask_idx = inpainting_test( 
                        ema_noise_pred_net, noise_scheduler, obs_cond, noisy_action, num_diffusion_iters
                    )
                
                    naction = action_opt
                    # to tensor # reshape back
                    naction = torch.tensor(naction, dtype=torch.float32).to(device)
                    naction = naction.unsqueeze(0)

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, _, info = env.step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                actions.append(normalize_data(action[i], stats=stats['action']) )
                imgs.append(env.render(mode='rgb_array'))

                # save_action_image(action[i], env.render(mode='rgb_array'), step_idx)

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    # print out the maximum target coverage
    print('Score: ', max(rewards))

    # visualize
    # from IPython.display import Video
    # vwrite('vis.mp4', imgs)
    # Video('vis.mp4', embed=True, width=256, height=256)

    save_video('vis2.mp4', imgs)

    # save actions
    np.save('actions.npy', np.array(actions))

def save_video(file_name, frames, fps=20, video_format='mp4'):
    import skvideo.io
    skvideo.io.vwrite(
        file_name,
        frames,
        inputdict={
            '-r': str(int(fps)),
        },
        outputdict={
            '-f': video_format,
            '-pix_fmt': 'yuv420p', # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
        }
    )

def save_action_image(action, image, idx):
    # create folder for images
    if not os.path.exists('images'):
        os.makedirs('images')

    # save image, action as image for visualization
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(f'images/img_{idx}_{action}.png')


def vis_actions(actions):
    plt.figure()
    plt.plot(actions[:,0], actions[:,1], label='trajectory')
    plt.scatter(actions[:,0], actions[:,1], c=range(len(actions)), label='time')
    plt.colorbar()
    plt.legend()
    plt.axis('equal')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.show()

if __name__ == '__main__':
    args = get_args()
    env = create_env()
    dataloader = creat_dataset(args)
    noise_pred_net, noise_scheduler = create_network(args)

    args.mode = 'eval'

    if args.mode == 'train':
        train(args, noise_pred_net, noise_scheduler, dataloader)
        print("Training done!")
    elif args.mode == 'eval':
        load_modal(args, noise_pred_net)
        eval(args, env, noise_pred_net, noise_scheduler, dataloader.dataset.stats)
        print("Evaluation done!")

