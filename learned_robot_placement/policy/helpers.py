import torch.nn as nn

def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims

    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        # nn.init.orthogonal(layers[-1].weight)
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
            # layers.append(nn.LeakyReLU(negative_slope=0.1))
    net = nn.Sequential(*layers)
    return net