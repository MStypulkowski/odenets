import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import seed_everything
from models import ODEOptimizer

print(f'Available GPUs: {torch.cuda.device_count()}\n')


def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2


if __name__ == '__main__':
    seed = 42
    n_epochs = 5000
    lr = 1e-4
    result_dir = '/pio/scratch/1/mstyp/odenets/results'
    t1 = 10
    dynamic_type = 'nn'
    hid_dim = 64
    n_layers = 3
    # dynamic_type = 'linear'
    alpha1D = True

    if dynamic_type == 'nn':
        name = dynamic_type + '_' + str(n_layers) + '_' + str(hid_dim) + '_' + str(t1)
    elif dynamic_type == 'linear':
        name = dynamic_type + '_alpha1D_' + str(alpha1D) + '_' + str(t1)
    print(n_epochs, lr, name, '\n')

    seed_everything(seed)
    # w0 = torch.randn(1, 2).float().cuda()
    w0 = torch.tensor([[-0.5, 2.5]]).cuda()
    loss = rosenbrock

    ##############################
    # Get path of gradient descent
    ##############################
    print('Training using gradient descent...\n')
    w_param = torch.nn.Parameter(w0.clone())
    optimizer = torch.optim.Adam([w_param], lr=5e-1)

    w_path = w0.clone()
    
    for epoch in range(n_epochs - 1):
        optimizer.zero_grad()
        loss_val = loss(w_param[0, 0], w_param[0, 1])
        loss_val.backward()
        optimizer.step()
        if (epoch + 1) in [1, 2, 3, 5, 10, 50, 100, 500, 4999]:
            w_path = torch.cat([w_path, w_param.detach()], dim=0)
        if epoch % 500 == 0:
            print(f"Epoch: {epoch} Loss: {loss_val.item():.4f} X: {w_param[0, 0].item():.4f} Y: {w_param[0, 1].item():.4f}")
    print(f'\nPoint found: {w_param.cpu().detach().numpy()}\n\n')

    ##############################################
    # Force ODE Network to learn the gradient path
    ##############################################
    print('Forcing ODE Network to learn the path...\n')
    model = ODEOptimizer(2, w0=w0, dynamic_type=dynamic_type, hid_dim=hid_dim, alpha1D=alpha1D, n_layers=n_layers).cuda()
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.ode.parameters(), lr=lr)
    integration_times = torch.linspace(0, t1, len(w_path)).float().cuda()
    # integration_times = torch.nn.Parameter(torch.linspace(0, t1, len(w_path)).float().cuda())
    # optimizer = torch.optim.Adam(list(model.ode.parameters()) + list(integration_times), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        wt = model(integration_times=integration_times).squeeze(1)
        loss_val = loss(wt, w_path)
        loss_val.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch: {epoch} Loss: {loss_val.item():.4f}")

    #########################################
    # Plot both paths on Rosenbrock's contour
    #########################################
    x = np.arange(-1.5, 1.5, 0.1)
    y = np.arange(-1, 3, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    w_path = w_path.cpu()
    wt = wt.detach().cpu()

    plt.figure()
    plt.contour(X, Y, Z, 100)
    plt.plot(wt[:, 0], wt[:, 1], 'o-', c='red', label='ODE time')
    plt.plot(w_path[:, 0], w_path[:, 1], 'o-', c='orange', label='ODE iter')
    plt.legend()
    plt.colorbar()
    plt.title(name)
    plt.savefig(os.path.join(result_dir, 'path_forcing_rosenbrock_' + name + '.png'))
