import numpy as np
import cvxpy as cp

import torch


# ====================== create patch and masks ======================
def line_patch(x_length=16, patch_value=0, mask_index=np.arange(3, 7)):
    patch = np.zeros((x_length, 1))
    mask = np.zeros((x_length, 1))

    patch[mask_index] = patch_value
    mask[mask_index] = 1

    return patch, mask


# ====================== create inpainting methods ======================
def vanila_inpainting(x_ori, mask, patch):
    
        '''
        x_ori: (batch_size, x_length)
        '''
    
        # if x is tensor, detach and convert to numpy
        if isinstance(x_ori, torch.Tensor):
            x = x_ori.detach().cpu().numpy()
        else:
            x = x_ori
    
        # make mask and patch have same shape as x, they are at the second dimension
        mask_batch = np.tile(mask, (x.shape[0], 1, 1))
        patch_batch = np.tile(patch, (x.shape[0], 1, 1))
    
        assert x.shape == mask_batch.shape == patch_batch.shape
    
        # inpainting
        x_hat = x * (1 - mask_batch) + patch_batch * mask_batch
    
        # convert to tensor if x_ori is tensor
        if isinstance(x_ori, torch.Tensor):
            x_hat = torch.tensor(x_hat, dtype=torch.float32).to(x_ori.device)
    
        return x_hat


def MSE_opt(x_ori, mask, patch, w):

    '''
    x_ori: (batch_size, x_length)
    '''

    # if x is tensor, detach and convert to numpy
    if isinstance(x_ori, torch.Tensor):
        x = x_ori.detach().cpu().numpy()
    else:
        x = x_ori


    # make mask and patch have same shape as x, they are at the second dimension
    mask_batch = np.tile(mask, (x.shape[0], 1, 1))
    patch_batch = np.tile(patch, (x.shape[0], 1, 1))

    assert x.shape == mask_batch.shape == patch_batch.shape

    # flatten the second and third dimensions
    x_flat = x.reshape(x.shape[0], -1)
    mask_flat = mask_batch.reshape(mask_batch.shape[0], -1)
    patch_flat = patch_batch.reshape(patch_batch.shape[0], -1)

    # create optimization variable
    x_hat = cp.Variable(x_flat.shape)

    # create optimization problem
    # loss 1: inpainting loss
    masked_diff = cp.multiply(mask_flat, (x_hat - patch_flat))
    mse_loss_mask = cp.sum_squares(masked_diff)

    # loss 2: in-dist loss
    backward_diff = x_hat - x_flat
    mse_loss_backward = cp.sum_squares(backward_diff)

    # total loss with weight
    total_loss = mse_loss_mask + w * mse_loss_backward

    # create optimization problem and solve
    problem = cp.Problem(cp.Minimize(total_loss))
    problem.solve()

    # get optimized results
    x_hat = x_hat.value

    '''
    # calculate loss again
    masked_diff = np.multiply(mask_flat, (x_hat - patch_flat))
    mse_loss_mask = np.mean(masked_diff**2)
    backward_diff = x_hat - x_flat
    mse_loss_backward = np.mean(backward_diff**2)

    print(f'loss: {mse_loss_mask}')
    print(f'loss: {mse_loss_backward}')

    # reshape to original shape
    x_hat = x_hat.reshape(x.shape)
    '''

    # convert to tensor if x_ori is tensor
    if isinstance(x_ori, torch.Tensor):
        x_hat = torch.tensor(x_hat, dtype=torch.float32).to(x_ori.device)
    
    return x_hat


def MSE_inequ_opt(x_ori, mask, patch, threshold):
    '''
    x_ori: (batch_size, x_length)
    '''

    # if x is tensor, detach and convert to numpy
    if isinstance(x_ori, torch.Tensor):
        x = x_ori.detach().cpu().numpy()
    else:
        x = x_ori


    # make mask and patch have same shape as x, they are at the second dimension
    mask_batch = np.tile(mask, (x.shape[0], 1, 1))
    patch_batch = np.tile(patch, (x.shape[0], 1, 1))

    batch_size = x.shape[0]
    action_length = x.shape[1]
    num_non_zero = np.sum(mask) # number of non zero elements in mask

    assert x.shape == mask_batch.shape == patch_batch.shape

    # flatten the second and third dimensions
    x_flat = x.reshape(x.shape[0], -1)
    mask_flat = mask_batch.reshape(mask_batch.shape[0], -1)
    patch_flat = patch_batch.reshape(patch_batch.shape[0], -1)

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

    print(f'loss: {mse_loss_mask}')
    print(f'loss: {mse_loss_backward}')
    '''

    # reshape to original shape
    x_hat = x_hat.reshape(x.shape)

    # convert to tensor if x_ori is tensor
    if isinstance(x_ori, torch.Tensor):
        x_hat = torch.tensor(x_hat, dtype=torch.float32).to(x_ori.device)
    
    return x_hat