import torch
import torch.nn as nn
from models.ode_essentials import ODEBlock, ODEfunc, LinearDynamic


class LogisticODE(nn.Module):
    def __init__(self, w0=None, dynamic_type='nn', hid_dim=64, alpha1D=True, rtol=1e-3, atol=1e-3):
        super(LogisticODE, self).__init__()
        if dynamic_type == 'nn':
            self.ode_func = ODEfunc(3, hid_dim)
        elif dynamic_type == 'linear':
            self.ode_func = LinearDynamic(3, alpha1D=alpha1D)
        self.ode = ODEBlock(self.ode_func, rtol=rtol, atol=atol)
        self.activation = nn.Sigmoid()
        if w0 is None:
            self.w0 = torch.randn(1, 3).float().cuda()
        else:
            self.w0 = w0.clone()

    def forward(self, x, integration_times=None, regularization=False):
        wt = self.ode(self.w0, integration_times=integration_times)
        if integration_times is None:
            wt = wt[0]
            y = torch.matmul(torch.cat((x, torch.ones((x.shape[0], 1)).cuda()), axis=1), wt.T)
            if not regularization:
                return self.activation(y)
            return self.activation(y), (wt ** 2).sum()
        return wt

    def get_probs(self, wt, x):
        y = torch.matmul(torch.cat((x, torch.ones((x.shape[0], 1)).cuda()), axis=1), wt.T)
        return self.activation(y)


class SimpleLogisticODE(nn.Module):
    def __init__(self, hid_dim):
        super(SimpleLogisticODE, self).__init__()
        self.ode_func = ODEfunc(2, hid_dim)
        self.ode = ODEBlock(self.ode_func)
        self.activation = nn.Sigmoid()
        # self.w0 = torch.zeros(1, 2).float().cuda()
        self.w0 = torch.randn(1, 2).float().cuda()

    def forward(self, x, integration_times=None, reg=False):
        wt = self.ode(self.w0, integration_times=integration_times)
        y = x @ wt.T
        if not reg:
            return self.activation(y)
        return self.activation(y), (wt ** 2).sum()


class ODEOptimizer(nn.Module):
    def __init__(self, in_dim, conditional=False, layer_name='concat', w0=None, dynamic_type='nn', hid_dim=64, alpha1D=True, n_layers=2, rtol=1e-3, atol=1e-3, dw_dims=None):
        super(ODEOptimizer, self).__init__()
        self.conditional = conditional
        if dynamic_type == 'nn':
            self.ode_func = ODEfunc(in_dim, hid_dim, n_layers=n_layers, conditional=conditional, layer_name=layer_name, dw_dims=dw_dims)
        elif dynamic_type == 'linear':
            self.ode_func = LinearDynamic(in_dim, alpha1D=alpha1D)
        self.ode = ODEBlock(self.ode_func, rtol=rtol, atol=atol)
        if w0 is None:
            self.w0 = torch.randn(1, in_dim).float().cuda()
        else:
            self.w0 = w0.clone()

    def forward(self, w0=None, dw=None, integration_times=None):
        if w0 is None:
            w0 = self.w0
        wt = self.ode(w0, dw=dw, integration_times=integration_times)
        if integration_times is None:
            if self.conditional:
                return wt[0][-1] # wt is in the form (dwdt, dLdw), where dw has shape [n_integration_times, bsz, w_dim]
            return wt[-1]
        if self.conditional:
            return wt[0][1:]
        return wt[1:]


class SoftmaxODE(nn.Module):
    def __init__(self, layer_dims, w0=None, dynamic_type='nn', hid_dim=2048, n_layers=4, alpha1D=True, conditional=True, data_dims=None, rtol=1e-3, atol=1e-3):
        super(SoftmaxODE, self).__init__()
        self.conditional = conditional
        self.layer_dims = layer_dims
        self.in_dim = layer_dims[0] * layer_dims[1] + layer_dims[1]
        if dynamic_type == 'nn':
            self.ode_func = ODEfunc(self.in_dim, hid_dim, n_layers=n_layers, conditional=conditional, data_dims=data_dims)
        elif dynamic_type == 'linear':
            self.ode_func = LinearDynamic(self.in_dim, alpha1D=alpha1D)
        self.ode = ODEBlock(self.ode_func, rtol=rtol, atol=atol)
        if w0 is None:
            self.w0 = torch.randn(1, self.in_dim).float().cuda()
        else:
            self.w0 = w0.clone()
        self.activation = nn.Softmax(dim=1)

    def forward(self, x, integration_times=None):
        if self.conditional:
            w0_epoch = self.w0.expand(x.shape[0], -1)
        else:
            w0_epoch = self.w0
        wt = self.ode(w0_epoch, x, integration_times=integration_times)
        if integration_times is None:
            if self.conditional:
                wt = wt[0][-1] # wt is in the form (dw, dx), where dw has shape [n_integration_times, bsz, w_dim]
                weights = wt[:, :self.layer_dims[0] * self.layer_dims[1]].reshape(-1, self.layer_dims[0], self.layer_dims[1])
                biases = wt[:, -self.layer_dims[1]:].reshape(-1, self.layer_dims[1])
                y = torch.bmm(x.unsqueeze(1), weights).squeeze(1) + biases
            else:
                wt = wt[-1]
                weights = wt[:, :self.layer_dims[0] * self.layer_dims[1]].reshape(self.layer_dims[0], self.layer_dims[1])
                biases = wt[:, -self.layer_dims[1]:]
                y = x @ weights + biases
            return self.activation(y)
        return wt
        