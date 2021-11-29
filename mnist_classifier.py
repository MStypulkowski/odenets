import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm

from models import SoftmaxReg, SoftmaxODE
from utils import seed_everything

print(f'Available GPUs: {torch.cuda.device_count()}\n')


if __name__ == '__main__':
    seed = 42
    n_epochs = 1000
    lr = 1e-5
    bsz = 2048
    data_dir = '/pio/scratch/1/mstyp/odenets/datasets'
    result_dir = '/pio/scratch/1/mstyp/odenets/results'

    # model_name = 'SoftmaxReg'
    model_name = 'SoftmaxODE'
    data_dependent = False
    data_dims = (784, 64)
    
    layer_dims=(784, 10)

    dynamic_type = 'nn'
    hid_dim = 1024
    n_layers = 4

    # dynamic_type = 'linear'
    alpha1D = False

    seed_everything(seed)

    mnist_train = MNIST(data_dir, train=True, transform=Compose([ToTensor()]), download=True)
    mnist_test = MNIST(data_dir, train=False, transform=Compose([ToTensor()]), download=True)
    train_dl = DataLoader(mnist_train, batch_size=bsz, shuffle=True)
    test_dl = DataLoader(mnist_test, batch_size=bsz, shuffle=True)

    model = {
        'SoftmaxReg': SoftmaxReg(),
        'SoftmaxODE': SoftmaxODE(layer_dims, hid_dim=hid_dim, n_layers=n_layers, data_dependent=data_dependent, data_dims=data_dims)
    }[model_name].cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    # print(model)
    loss = nn.NLLLoss()

    for epoch in range(n_epochs):
        for x, y in tqdm(train_dl, desc=f'Epoch: {epoch}'):
            x = x.reshape(-1, 784).cuda()
            y = y.cuda()
            probs = model(x)
            loss_val = loss(probs, y)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
        if epoch % 1 == 0:
            correct = 0.
            total = 0.
            model.eval()
            with torch.no_grad():
                for x, y in test_dl:
                    x = x.reshape(-1, 784).cuda()
                    y = y.cuda()
                    y_pred = torch.argmax(model(x), dim=1)
                    correct += sum(y_pred == y)
                    total += len(y)

            print(f"Epoch: {epoch} Loss: {loss_val.item():.4f} Accuacy: {100 * correct / total :.2f}%")