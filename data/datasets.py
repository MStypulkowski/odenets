import numpy as np
import torch


def get_toy_data():
    m1 = [-2, 0]
    cov1 = [[1, -2], [-2, 5]]

    m2 = [1, -1]
    cov2 = [[3, 2], [2, 2]]

    x1, y1 = np.random.multivariate_normal(m1, cov1, 500).T
    x2, y2 = np.random.multivariate_normal(m2, cov2, 500).T

    x = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([y1, y2], axis=0)

    target1 = np.ones((500,))
    target2 = np.zeros((500,))

    target = np.concatenate([target1, target2], axis=0)

    X = torch.from_numpy(np.concatenate([np.expand_dims(x,1) ,np.expand_dims(y,1)], axis=1)).float()
    Y = torch.from_numpy(target).reshape(-1,1).float()

    return X, Y