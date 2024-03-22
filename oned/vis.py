from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

def vis_1d(samples,):
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
