from scipy.stats import gaussian_kde
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

# calculate losses and matrics
def calculate_mse_loss(x_results, mask, patch):
    # get the last action in every batch
    last_action = x_results[:, -1, :, :] # batch size, diff steps, length, action dim
    last_action = last_action.squeeze() 

    # calculate mse loss with the patch value
    mean = np.mean(last_action, axis=0)
    mse_loss = np.mean(np.square(mean - patch))
    return mse_loss

# visualize dist
def vis_last_action(x_results, mask, patch, ax):
    # get the last action in every batch
    last_action = x_results[:, -1, :, :] # batch size, diff steps(remove), length, action dim
    last_action = last_action.squeeze()

    # calculate mean and std
    action_length = last_action.shape[1]
    mean = np.mean(last_action, axis=0)
    std = np.std(last_action, axis=0)

    # get mask index
    mask_1d = mask.squeeze()
    patch_1d = patch.squeeze()
    mask_index = np.where(mask_1d == 1)[0]


    I = np.arange(action_length)
    # plot
    ax.plot(I, mean, label='mean')
    ax.fill_between(I, mean-std, mean+std, alpha=0.3)
    ax.plot(mask_index, patch_1d[mask_index], label='patch_value', color='r', marker='o')

    ax.set_xlabel('Action Length')
    ax.set_ylabel('Value')
    ax.set_title('Mean and Std over Action Length')
    ax.legend()
    


def vis_diff_steps(x_results, mask, patch, ax):
    # get actions over diff steps
    diff_action = np.mean(x_results, axis=2) # batch size, diff steps, length
    diff_action = diff_action.squeeze()

    # calculate mean and std
    diff_steps = diff_action.shape[1]
    mean = np.mean(diff_action, axis=0)
    std = np.std(diff_action, axis=0)

    I = np.arange(diff_steps)
    # plot
    ax.plot(I, mean, label='mean')
    ax.fill_between(I, mean-std, mean+std, alpha=0.3)
    ax.set_xlabel('Diffusion Steps')
    ax.set_ylabel('Value')
    ax.set_title('Mean and Std over Diffusion Steps')
    ax.legend()




def vis_1d(samples):
    # samples: [Batch, Length]
    samples_reshaped = samples.reshape(-1, 1)  # Reshape to a 2D array for KDE
    B, L = samples.shape
    dimensions = np.repeat(np.arange(L), B).reshape(-1, 1)  

    # Combine samples and dimensions for KDE input
    kde_input = np.hstack((dimensions, samples_reshaped))


    # '''
    # Create a gaussian_kde object
    kde = gaussian_kde(kde_input.T)

    # Create a grid of points for evaluating the KDE
    dim_grid = np.repeat(np.arange(L), L)
    value_grid = np.linspace(samples.min(), samples.max(), 100)
    dim_grid, value_grid = np.meshgrid(dim_grid, value_grid)
    kde_grid = np.vstack([dim_grid.ravel(), value_grid.ravel()])

    # Evaluate the KDE on the grid
    Z = kde(kde_grid).reshape(dim_grid.shape)

    # Plot the result using pcolormesh

    plt.figure(figsize=(10, 8))
    plt.pcolormesh(dim_grid, value_grid, Z, shading='auto')
    plt.colorbar(label='Density')
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.title('2D Gaussian KDE of Dataset')
    plt.show()

    return kde


def do_vis(file_name):
    data = pickle.load(open(file_name, 'rb'))
    x_results = data['x_results']
    inpainting_dict = data['inpainting_dict']
    mask = inpainting_dict['mask']
    patch = inpainting_dict['patch']

    # create a sub folder
    sub_folder = file_name.split('/')[1]
    # create fig foler
    fig_folder = './{}/figs'.format(sub_folder)
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    # get file name from
    fig_save_name = fig_folder + "/" + file_name.split('/')[2] + '.png'
    

    fig, ax = plt.subplots(1, 2, figsize=(20,7))
    vis_last_action(x_results, mask, patch, ax[0])
    vis_diff_steps(x_results, mask, patch, ax[1])
    plt.savefig(fig_save_name)
    # plt.show()
    plt.close()

if __name__ == "__main__":
    file_name = './xxx/results_last.pkl'
    do_vis(file_name)