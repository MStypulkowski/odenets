import torch
import torch.nn as nn
from models.ode_utils import ODEBlock, ODEfunc


class Logistic(nn.Module):
    def __init__(self):
        super(Logistic, self).__init__()
        self.fc1 = nn.Linear(2, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.fc1(x))


class LogisticODE(nn.Module):
    def __init__(self, w0=None, hid_dim=64, rtol=1e-3, atol=1e-3):
        super(LogisticODE, self).__init__()
        self.ode_func = ODEfunc(3, hid_dim)
        self.ode = ODEBlock(self.ode_func, rtol=rtol, atol=atol)
        self.activation = nn.Sigmoid()
        if w0 is None:
            self.w0 = torch.randn(1, 3).float().cuda()
        else:
            self.w0 = w0.clone()

    def forward(self, x, integration_times=None, reg=False):
        wt = self.ode(self.w0, integration_times=integration_times)
        if integration_times is None:
            wt = wt[0]
            y = torch.matmul(torch.cat((x, torch.ones((x.shape[0], 1)).cuda()), axis=1), wt.T)
            if not reg:
                return self.activation(y)
            return self.activation(y), (wt ** 2).sum()
        return wt


class SimpleLogistic(nn.Module):
    def __init__(self):
        super(SimpleLogistic, self).__init__()
        self.fc1 = nn.Linear(2, 1, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.fc1(x))


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