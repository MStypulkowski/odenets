import torch
import torch.nn as nn


class ConcatLinear(nn.Module):
    def __init__(self, in_dim, out_dim, activation=True, data_dims=None):
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
    def __init__(self, in_dim, out_dim, activation=True, data_dims=None):
        super(ConcatLinear3D, self).__init__()
        x_in_dim, x_out_dim = data_dims

        self.fc_x = nn.Linear(x_in_dim, x_out_dim)
        self.fc_twx = nn.Linear(in_dim + x_out_dim + 1, out_dim)
        self.x_activation = nn.Softplus()
        self.activation = nn.Softplus() if activation else None
        
        # self.ones = nn.Parameter(torch.ones(1, 1), requires_grad=False)

    def forward(self, t, w, x):
        # print('Concat layer', t.shape, w.shape, x.shape)
        x = self.x_activation(self.fc_x(x))#.mean(0).view(1, -1)
        # ones = torch.ones(w.size(0), 1).to(w)
        # ones = self.ones.expand(w.size(0), 1)
        # t = ones * t
        t = torch.ones(w.size(0), 1).to(w) * t.clone().detach().requires_grad_(True).type_as(w)
        # print('Concat layer', t.shape, w.shape, x.shape)
        twx = torch.cat([t, w, x], dim=1)
        twx = self.fc_twx(twx)
        if self.activation:
            twx = self.activation(twx)
        return twx