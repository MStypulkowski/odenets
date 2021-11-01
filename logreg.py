import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from utils import seed_everything
from data import get_toy_data
from models import Logistic, LogisticODE

print(f'Available GPUs: {torch.cuda.device_count()}\n')


if __name__ == '__main__':
    seed = 42
    n_epochs = 10000
    lr = 1e-2
    result_dir = '/pio/scratch/1/mstyp/odenets/results'
    # dynamic_type = 'nn'
    hid_dim = 64
    dynamic_type = 'linear'
    alpha1D = False

    if dynamic_type == 'nn':
        name = dynamic_type + '_' + str(hid_dim)
    elif dynamic_type == 'linear':
        name = dynamic_type + '_alpha1D_' + str(alpha1D)
    print(n_epochs, lr, name, '\n')

    seed_everything(seed)
    w0 = torch.randn(1, 3).float().cuda()

    X, Y  = get_toy_data()
    X, Y = X.cuda(), Y.cuda()
    
    ######################################
    # LOGISTIC REGRESSION
    ######################################
    print('Training logistic regression...')
    model_logistic = Logistic().cuda()

    # initialize parameters so they match ODE models
    model_logistic.fc1.weight.data = w0[:, :2].clone()
    model_logistic.fc1.bias.data = w0[:, -1].clone()

    optimizer = torch.optim.Adam(model_logistic.parameters(), lr=lr)

    model_logistic.train()
    loss = nn.BCELoss()

    loss_vals = []
    ws = []
    bs = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model_logistic(X)
        loss_val = loss(y_pred, Y)
        loss_val.backward()
        optimizer.step()
        
        loss_vals.append(loss_val.item())
        ws.append(model_logistic.fc1.weight.detach().cpu().numpy()[0])
        bs.append(model_logistic.fc1.bias.detach().cpu().numpy()[0])
        
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch} Loss: {loss_val.item():.4f}")

    ws = np.array(ws)
    bs = np.array(bs)

    ######################################
    # ODE LOGISTIC REGRESSION
    ######################################
    print('\nTraining ODE logistic regression...')
    model_logisticODE = LogisticODE(w0=w0, dynamic_type=dynamic_type, hid_dim=hid_dim, alpha1D=alpha1D).cuda()

    optimizer = torch.optim.Adam(model_logisticODE.ode.parameters(), lr=lr)

    model_logisticODE.train()
    loss = nn.BCELoss()

    for epoch in range(n_epochs//10):
        optimizer.zero_grad()
        probs = model_logisticODE(X)
        loss_val = loss(probs, Y)
        loss_val.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} Loss: {loss_val.item():.4f}")

    # get parameters from multiple timestamps
    loss_vals_ode = []
    model_logisticODE.eval()
    with torch.no_grad():
        integration_times = torch.linspace(0, 1, n_epochs).float().cuda()
        ws_ode = model_logisticODE(None, integration_times=integration_times).detach().squeeze(1)
        ws_ode = torch.cat([w0, ws_ode], dim=0)
        
        for w_ode in ws_ode:
            probs = model_logisticODE.get_probs(w_ode.view(1, -1), X)
            loss_vals_ode.append(loss(probs, Y).item())
        
        ws_ode = ws_ode.cpu()

    x = np.arange(n_epochs)

    if dynamic_type == 'linear':
        w1 = model_logisticODE.ode_func.w1.data.cpu()
        alpha = model_logisticODE.ode_func.alpha.data.cpu()
        print(alpha, w1)
        ws_ode_true = (w0.cpu() - w1) * torch.exp(-alpha * x.reshape(-1, 1) / n_epochs) + w1

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    ax[0, 0].plot(x, loss_vals)
    ax[0, 0].plot(x, loss_vals_ode)
    ax[0, 0].set_xlabel('epoch/time')
    ax[0, 0].set_ylabel('loss')

    ax[0, 1].plot(x, bs, label='LogReg')
    ax[0, 1].plot(x, ws_ode[:, -1], label='ODE')
    if dynamic_type == 'linear':
        ax[0, 1].plot(x, ws_ode_true[:, -1], '--', label='GT')
        ax[0, 1].scatter(x[-1], w1[0, -1], s=100, marker='*', c='red', label='w_t1')
    ax[0, 1].set_xlabel('epoch/time')
    ax[0, 1].set_ylabel('b')
    ax[0, 1].legend()

    ax[1, 0].plot(x, ws[:, 0])
    ax[1, 0].plot(x, ws_ode[:, 0])
    if dynamic_type == 'linear':
        ax[1, 0].plot(x, ws_ode_true[:, 0], '--')
        ax[1, 0].scatter(x[-1], w1[0, 0], s=100, marker='*', c='red')
    ax[1, 0].set_xlabel('epoch/time')
    ax[1, 0].set_ylabel('w0')

    ax[1, 1].plot(x, ws[:, 1])
    ax[1, 1].plot(x, ws_ode[:, 1])
    if dynamic_type == 'linear':
        ax[1, 1].plot(x, ws_ode_true[:, 1], '--')
        ax[1, 1].scatter(x[-1], w1[0, 1], s=100, marker='*', c='red')
    ax[1, 1].set_xlabel('epoch/time')
    ax[1, 1].set_ylabel('w1')

    plt.suptitle(name, fontsize=20)
    plt.savefig(os.path.join(result_dir, 'logreg_params_' + name +'.png'))