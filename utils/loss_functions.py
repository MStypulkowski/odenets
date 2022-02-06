import torch
import numpy as np


def recall_loss(predictions, targets, weights):
    return torch.sum(weights * (predictions - targets) ** 2)


def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2


def sincos(x, y):
    return 20 * torch.sin(np.pi * x / 2) * torch.cos(np.pi * y / 2 - 0.5) + (x - 1)**2 + (y - 1)**2 


FUNCTIONS = {
    1: [rosenbrock, torch.tensor([[-0.5, 2.5]])],
    2: [sincos, torch.tensor([[-0.5, 2.5]])],
    3: []
}


def get_loss_function(id):
    return FUNCTIONS[id]