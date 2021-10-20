import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import seed_everything
from models import ODEOptimizer

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

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        wt = model()
        loss_val = loss(wt[0, 0], wt[0, 1])
        loss_val.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch: {epoch} Loss: {loss_val.item():.4f} W0: {wt[0, 0].item():.4f} W1: {wt[0, 0].item():.4f}")

    model.eval()
    with torch.no_grad():
        integration_times = torch.linspace(0, 1, 100).float().cuda()
        wts = model(integration_times=integration_times).detach().cpu().squeeze(1)
        wts = torch.cat([w0.cpu(), wts], dim=0)

    print(f'Point found: {wts[-1, :].numpy()}')
    x = np.arange(-1, 1.5, 0.1)
    y = np.arange(-1, 3, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)

    plt.contour(X, Y, Z, 100)
    plt.plot(wts[:, 0], wts[:, 1], c='red')
    plt.colorbar()
    plt.savefig(os.path.join(result_dir, 'rosenbrock.png'))