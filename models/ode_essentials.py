import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from models.ode_layers import ConcatLinear, ConcatLinear3D


class ODEfunc(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers=2, data_dependent=True, data_dims=None):
        super(ODEfunc, self).__init__()
        assert n_layers >= 2

        self.data_dependent = data_dependent
        if data_dependent:
            base_layer = ConcatLinear3D
        else:
            base_layer = ConcatLinear

        self.fcs = []
        self.fcs.append(base_layer(in_dim, hid_dim, data_dims=data_dims))
        for i in range(n_layers-2):
            self.fcs.append(base_layer(hid_dim, hid_dim, data_dims=data_dims))
        self.fcs.append(base_layer(hid_dim, in_dim, activation=False, data_dims=data_dims))
        # self.fcs.append(base_layer(hid_dim, in_dim + data_dims[1], activation=False, data_dims=data_dims))
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, t, inputs):
        if self.data_dependent:
            w, x = inputs
            # print('ODEfunc', t.shape, w.shape, x.shape)
            for fc in self.fcs:
                w = fc(t, w, x)
            return w, torch.zeros_like(x)
        else:
            w = inputs
            for fc in self.fcs:
                w = fc(t, w)
            return w


class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=1e-3, atol=1e-3, integration_times=torch.tensor([0, 1]).float()):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_times = integration_times
        self.rtol = rtol
        self.atol = atol

    def forward(self, w0, x, integration_times=None):
        if integration_times is None:
            integration_times = self.integration_times
        if self.odefunc.data_dependent:
            # print('ODEBlock', w0.shape, x.shape, integration_times.shape)
            inputs = (w0, x)
        else:
            inputs = w0
        # print('inputs', inputs.shape)
        wt = odeint(self.odefunc, inputs, integration_times, rtol=self.rtol, atol=self.atol)
        # print(wt.shape)
        # print('Final', len(wt), wt[0][1:].shape)
        return wt


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