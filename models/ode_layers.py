import torch
import torch.nn as nn


class ConcatLinear(nn.Module):
    def __init__(self, in_dim, out_dim, activation=True, dw_dims=None):
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


class ConcatLinear3D(nn.Module):
    def __init__(self, in_dim, out_dim, activation=True, dw_dims=None):
        super(ConcatLinear3D, self).__init__()
        dw_in_dim, dw_out_dim = dw_dims
        self.fc_dw = nn.Linear(dw_in_dim, dw_out_dim)
        self.fc_twdw = nn.Linear(in_dim + dw_out_dim + 1, out_dim)
        self.dw_activation = nn.Sigmoid()
        self.activation = nn.Softplus() if activation else None

    def forward(self, t, w, dw):
        dw = self.dw_activation(self.fc_dw(dw))
        t = torch.ones(w.size(0), 1).to(w) * t.clone().detach().requires_grad_(True).type_as(w)
        twdw = torch.cat([t, w, dw], dim=1)
        twdw = self.fc_twdw(twdw)
        if self.activation:
            twdw = self.activation(twdw)
        return twdw