import torch.nn as nn


class Logistic(nn.Module):
    def __init__(self):
        super(Logistic, self).__init__()
        self.fc1 = nn.Linear(2, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.fc1(x))


class SimpleLogistic(nn.Module):
    def __init__(self):
        super(SimpleLogistic, self).__init__()
        self.fc1 = nn.Linear(2, 1, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.fc1(x))


class SoftmaxReg(nn.Module):
    def __init__(self, in_dim=784, out_dim=10):
        super(SoftmaxReg, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        return self.activation(self.fc1(x))