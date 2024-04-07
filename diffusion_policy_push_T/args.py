import argparse

def get_args():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Noise Prediction Network')
    # choose mode, train or eval
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='mode')
    # inpainting method
    


    parser.add_argument('--pred_horizon', type=int, default=16, help='prediction horizon')
    parser.add_argument('--obs_horizon', type=int, default=2, help='observation horizon')
    parser.add_argument('--action_horizon', type=int, default=8, help='action horizon')
    parser.add_argument('--obs_dim', type=int, default=5, help='observation dimension')
    parser.add_argument('--action_dim', type=int, default=2, help='action dimension')


    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--num_diffusion_iters', type=int, default=100, help='number of diffusion iterations')


    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--checkpoint_name', type=str, default='noise_pred_net.pth', help='checkpoint name')

    # return the arguments
    return parser.parse_args()