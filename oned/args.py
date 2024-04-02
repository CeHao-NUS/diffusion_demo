import argparse 


def get_args():
    parser = argparse.ArgumentParser(description='Diffusion Models')
    parser.add_argument('--num_diffusion_iters', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)

    parser.add_argument('--const_patch', type=float, default=0)
    parser.add_argument('--loss_weight', type=float, default=1)
    parser.add_argument('--threshold', type=float, default=0.001)


    parser.add_argument('--folder_name', type=str, default='xxx')
    return parser.parse_args()