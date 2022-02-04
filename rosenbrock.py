import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch

from utils import seed_everything
from models import ODEOptimizer

print(f'Available GPUs: {torch.cuda.device_count()}\n')


def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2


if __name__ == '__main__':
    #####################################################
    # ARGS
    #####################################################

    seed = 42
    n_epochs = 200
    optimizer_name = 'Adam' # 'Adam' or 'SGD'
    lr = 1e-4
    result_dir = '/pio/scratch/1/mstyp/odenets/results'
    conditional = False
    dynamic_type = 'nn'
    hid_dim = 64
    layer_name = 'concat' # 'concat' or 'concatsquash'
    dw_dims = (2, 2)
    # dynamic_type = 'linear'
    alpha1D = True

    progressive_learning = True
    t0 = 0
    tn = 10
    integration_step = (tn - t0) / n_epochs
    constant_t0 = True
    n_inner_epochs = 200
    recall_past_trajectory = True
    last_point_weight = 1.
    # lambda_ = 1e-1
    # n_points_to_recall = 

    #####################################################
    # TRAINING PREP
    #####################################################

    name = 'conditional_' if conditional else layer_name + '_'
    name += 'progressive_' if progressive_learning else ''
    name += 'constant_' if constant_t0 else 'stepping_'
    name += 'recall_' + 'last_weight_' + str(last_point_weight) + '_' if recall_past_trajectory else ''
    if dynamic_type == 'nn':
        name += dynamic_type + '_' + str(hid_dim)
    elif dynamic_type == 'linear':
        name += dynamic_type + '_alpha1D_' + str(alpha1D)
    name += '_e' + str(n_epochs) + '_ie' + str(n_inner_epochs) + '_lr' + str(lr) + '_' + optimizer_name + '_tn' + str(tn)
    print(name, '\n')

    seed_everything(seed)
    # w0 = torch.randn(1, 2).float().cuda()
    w0 = torch.tensor([[-0.5, 2.5]]).cuda()

    model = ODEOptimizer(2, conditional=conditional, layer_name=layer_name, w0=w0, dynamic_type=dynamic_type, hid_dim=hid_dim, alpha1D=alpha1D, dw_dims=dw_dims).cuda()
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.ode.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.ode.parameters(), lr=lr)
    loss = rosenbrock

    wts_iter = [w0[0].detach().cpu().numpy()]
    rosenbrock_vals_iter = []

    model.train()
    wt = w0.clone() if conditional else None
    # dw/dt = f(w, t, dL/dw, theta)
    
    curr_integration_times = torch.tensor([t0, t0 + integration_step]).float().cuda() if progressive_learning else None
    curr_w0 = None if constant_t0 else w0.clone()
    past_trajectory = None
    iter_history = []
    ode_history = []
    # recall_loss = torch.nn.MSELoss()

    def recall_loss(predictions, targets, weights):
        return torch.sum(weights * (predictions - targets) ** 2)

    #####################################################
    # TRAINING
    #####################################################

    for epoch in range(n_epochs):
        if recall_past_trajectory:
            past_integration_times = torch.arange(t0, t0 + epoch * integration_step + 1e-12, integration_step).float().cuda()
            if len(past_integration_times) >= 2:
                model.eval()
                with torch.no_grad():
                    past_trajectory = model(w0=curr_w0, integration_times=past_integration_times).squeeze(1)
                model.train()
            else:
                past_trajectory = None

        if conditional:
            optimizer.zero_grad()
            wt = wt.requires_grad_(True)
            loss_val_dw = loss(wt[0, 0], wt[0, 1])
            loss_val_dw.backward()
            dw = wt.grad
            wt = wt.detach()
        else:
            dw = None
        
        optimizer.zero_grad()
        wt = model(w0=curr_w0, dw=dw, integration_times=curr_integration_times)

        if progressive_learning:
            wt = wt[-1]

        loss_val = loss(wt[-1, 0], wt[-1, 1])

        loss_val.backward()
        optimizer.step()
        wt = wt.detach()

        wts_iter.append(wt[-1].cpu().numpy())
        rosenbrock_vals_iter.append(loss_val.item())

        if recall_past_trajectory:
            past_trajectory = wt if past_trajectory is None else torch.cat([past_trajectory, wt], dim=0)
            past_integration_times = torch.cat([past_integration_times, torch.tensor(t0 + (epoch + 1) * integration_step).unsqueeze(0).cuda()])

            for inner_epoch in range(n_inner_epochs):
                if conditional:
                    optimizer.zero_grad()
                    wt = wt.requires_grad_(True)
                    loss_val_dw = loss(wt[0, 0], wt[0, 1])
                    loss_val_dw.backward()
                    dw = wt.grad
                    wt = wt.detach()
                else:
                    dw = None
                
                optimizer.zero_grad()
                wt = model(w0=curr_w0, dw=dw, integration_times=past_integration_times).squeeze(1)

                weights = torch.ones(wt.shape).cuda() * 0.1
                weights[-1] *= last_point_weight
                loss_recall = recall_loss(wt, past_trajectory, weights)

                loss_recall.backward()
                optimizer.step()
                wt = wt.detach()

                if epoch % 10 == 0:
                    iter_history.append(np.array(wts_iter))
                    ode_history.append(torch.cat([w0.clone(), wt], dim=0).cpu().numpy())

        if progressive_learning:
            if constant_t0:
                curr_integration_times[1] += integration_step
            else:
                curr_integration_times += integration_step
                curr_w0 = wt

        if epoch % 1 == 0:
            if recall_past_trajectory:
                print(f"Epoch: {epoch} Loss: {loss_val.item():.4f} X: {wts_iter[-1][0]:.4f} Y: {wts_iter[-1][1]:.4f} Loss recall: {loss_recall.item():.4f}, X adjusted: {wt[-1, 0].item():.4f}, Y adjusted: {wt[-1, 1].item():.4f}")
            else:
                print(f"Epoch: {epoch} Loss: {loss_val.item():.4f} X: {wts_iter[-1][0]:.4f} Y: {wts_iter[-1][1]:.4f}")

    wts_iter = np.array((wts_iter))

    #####################################################
    # EVALUATION
    #####################################################

    rosenbrock_vals = []
    model.eval()
    with torch.no_grad():
        integration_times = torch.linspace(t0, tn, n_epochs).float().cuda()
        wts = model(dw=dw, integration_times=integration_times).detach().cpu().squeeze(1)
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
    plt.plot(wts_iter[:, 0], wts_iter[:, 1], 'o-', c='orange', label='ODE iter')
    plt.plot(wts[:, 0], wts[:, 1], 'o-', c='red', label='ODE time')
    if dynamic_type == 'linear':
        plt.scatter(model.ode_func.w1.data.cpu()[0, 0], model.ode_func.w1.data.cpu()[0, 1], s=100, marker='*', c='black', label='w_t1')
    plt.legend()
    plt.colorbar()
    plt.title(name)
    plt.savefig(os.path.join(result_dir, 'rosenbrock_path_' + name + '.png'))

    if recall_past_trajectory:
        print("\nCreating GIF...")

        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1, 3)

            ax.contour(X, Y, Z, 100)
            ax.plot(iter_history[i][:, 0], iter_history[i][:, 1], 'o-', c='orange', label='ODE iter')
            ax.plot(ode_history[i][:, 0], ode_history[i][:, 1], 'o-', c='red', label='ODE time')
            
            ax.legend()

        ani = FuncAnimation(fig, animate, interval=40, repeat=True, frames=len(iter_history))
        ani.save(os.path.join(result_dir, 'gifs', 'rosenbrock_training_' + name + '.gif'), dpi=300, writer=PillowWriter(fps=25))

    # plt.figure()
    # plt.plot(np.arange(n_epochs), rosenbrock_vals_iter, c='orange', label='ODE iter')
    # plt.plot(np.arange(n_epochs), rosenbrock_vals, c='red', label='ODE time')
    # plt.legend()
    # plt.title(name)
    # plt.savefig(os.path.join(result_dir, 'rosenbrock_vals_' + name + '.png'))