import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class ConcatLinear(nn.Module):
    def __init__(self, in_dim, out_dim, activation=True):
        super(ConcatLinear, self).__init__()
        self.fc = nn.Linear(in_dim + 1, out_dim)
        self.activation = nn.Softplus() if activation else None
        self.ones = nn.Parameter(torch.ones(1, 1), requires_grad=False)

    def forward(self, t, w):
        ones = self.ones.expand(w.size(0), 1)
        t = ones * t
        tw = torch.cat([t, w], dim=1)
        tw = self.fc(tw)
        if self.activation:
            tw = self.activation(tw)
        return tw


class ODEfunc(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers=2):
        super(ODEfunc, self).__init__()
        assert n_layers >= 2
        self.fcs = []
        self.fcs.append(ConcatLinear(in_dim, hid_dim))
        for i in range(n_layers-2):
            self.fcs.append(ConcatLinear(hid_dim, hid_dim))
        self.fcs.append(ConcatLinear(hid_dim, in_dim, activation=False))
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, t, w):
        for fc in self.fcs:
            w = fc(t, w)
        return w


class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=1e-3, atol=1e-3, integration_times=torch.tensor([0, 1]).float()):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_times = nn.Parameter(integration_times, requires_grad=False)
        self.rtol = rtol
        self.atol = atol

    def forward(self, w0, integration_times=None):
        if integration_times is None:
            integration_times = self.integration_times
        wt = odeint(self.odefunc, w0, integration_times, rtol=self.rtol, atol=self.atol)
        return wt[1:]


class LinearDynamic(nn.Module):
    def __init__(self, in_dim, alpha1D=True):
        super(LinearDynamic, self).__init__()
        if alpha1D:
            self.alpha = nn.Parameter(torch.randn(1))
        else:
            self.alpha = nn.Parameter(torch.randn(1, in_dim))
        self.w1 = nn.Parameter(torch.randn(1, in_dim))

    def forward(self, t, w):
        return self.alpha * (self.w1 - w)