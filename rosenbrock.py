import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import seed_everything
from models import ODEOptimizer

print(f'Available GPUs: {torch.cuda.device_count()}')


def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2


if __name__ == '__main__':
    result_dir = '/pio/scratch/1/mstyp/odenets/results'
    seed = 42
    n_epochs = 3000

    seed_everything(seed)
    # w0 = torch.randn(1, 2).float().cuda()
    w0 = torch.tensor([[-0.5, 2.5]]).cuda()

    model = ODEOptimizer(2, w0=w0).cuda()
    optimizer = torch.optim.Adam(model.ode.parameters(), lr=0.001)
    loss = rosenbrock

    wts_iter = [w0[0].detach().cpu().numpy()]
    rosenbrock_vals_iter = []
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        wt = model()
        loss_val = loss(wt[0, 0], wt[0, 1])
        loss_val.backward()
        optimizer.step()

        wts_iter.append(wt[0].detach().cpu().numpy())
        rosenbrock_vals_iter.append(loss_val.item())

        if epoch % 500 == 0:
            print(f"Epoch: {epoch} Loss: {loss_val.item():.4f} X: {wt[0, 0].item():.4f} Y: {wt[0, 1].item():.4f}")
    wts_iter = np.array((wts_iter))

    rosenbrock_vals = []
    model.eval()
    with torch.no_grad():
        integration_times = torch.linspace(0, 1, n_epochs).float().cuda()
        wts = model(integration_times=integration_times).detach().cpu().squeeze(1)
        wts = torch.cat([w0.cpu(), wts], dim=0)
        for w in wts:
            rosenbrock_vals.append(rosenbrock(w[0], w[1]))

    print(f'\nPoint found: {wts[-1, :].numpy()}')
    x = np.arange(-1.5, 1.5, 0.1)
    y = np.arange(-1, 3, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    plt.figure()
    plt.contour(X, Y, Z, 100)
    plt.plot(wts[:, 0], wts[:, 1], c='red', label='ODE time')
    plt.plot(wts_iter[:, 0], wts_iter[:, 1], c='orange', label='ODE iter')
    plt.legend()
    plt.colorbar()
    plt.savefig(os.path.join(result_dir, 'rosenbrock_path.png'))

    plt.figure()
    plt.plot(np.arange(n_epochs), rosenbrock_vals, c='red', label='ODE time')
    plt.plot(np.arange(n_epochs), rosenbrock_vals_iter, c='orange', label='ODE iter')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'rosenbrock_vals.png'))