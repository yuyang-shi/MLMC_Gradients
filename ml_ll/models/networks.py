import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, net_dims, nonlinearity):
        """
        MLP module with 1 output.
        :param net_dims: list [input_dim, layer1, ..., output]
        :param nonlinearity: nonlinearity function
        """
        super(MLP, self).__init__()
        self.nonlinearity = nonlinearity
        self.fc_modules = nn.ModuleList()
        for in_dim, out_dim in zip(net_dims[:-2], net_dims[1:-1]):
            self.fc_modules.append(nn.Linear(in_dim, out_dim))
            self.fc_modules.append(nonlinearity)
        self.fc_modules.append(nn.Linear(net_dims[-2], net_dims[-1]))
        self.net_dims = net_dims

    def forward(self, x):
        for module in self.fc_modules:
            x = module(x)
        return [x]


class MLP_Normal(nn.Module):
    def __init__(self, net_dims, nonlinearity):
        """
        MLP module with 2 outputs, 2nd output is exponentiated.
        :param net_dims: list [input_dim, layer1, ..., output]
        :param nonlinearity: nonlinearity function
        """
        super(MLP_Normal, self).__init__()
        self.nonlinearity = nonlinearity
        self.fc_modules = nn.ModuleList()
        for in_dim, out_dim in zip(net_dims[:-2], net_dims[1:-1]):
            self.fc_modules.append(nn.Linear(in_dim, out_dim))
            self.fc_modules.append(nonlinearity)
        self.fc_modules.append(nn.Linear(net_dims[-2], net_dims[-1] * 2))
        self.net_dims = net_dims

    def forward(self, x):
        for module in self.fc_modules:
            x = module(x)
        mu, logsigma = torch.chunk(x, 2, dim=-1)
        sig = torch.exp(logsigma)
        return mu, sig


class MLP_Normal_Constant_Sigma(nn.Module):
    def __init__(self, net_dims, nonlinearity, init_logsigma, fix_logsigma):
        """
        MLP module for 1st output, scalar as expanded vector for 2nd output, 2nd output is exponentiated.
        :param net_dims: list [input_dim, layer1, ..., output], network dim for 1st output
        :param nonlinearity: nonlinearity function for first output
        :param init_logsigma: initialized value for logsigma
        :param fix_logsigma: whether the 2nd output is fixed or a parameter
        """
        super(MLP_Normal_Constant_Sigma, self).__init__()
        self.nonlinearity = nonlinearity
        self.fc_modules = nn.ModuleList()
        for in_dim, out_dim in zip(net_dims[:-2], net_dims[1:-1]):
            self.fc_modules.append(nn.Linear(in_dim, out_dim))
            self.fc_modules.append(nonlinearity)
        self.mu = nn.Linear(net_dims[-2], net_dims[-1])
        if fix_logsigma:
            self.register_buffer("logsigma", torch.tensor([init_logsigma], dtype=torch.float))
        else:
            self.logsigma = nn.Parameter(torch.tensor([init_logsigma], dtype=torch.float))
        self.net_dims = net_dims

    def forward(self, x):
        for module in self.fc_modules:
            x = module(x)
        mu = self.mu(x)
        return mu, torch.exp(self.logsigma.expand(self.net_dims[-1]))


class Linear_Normal(nn.Module):
    def __init__(self, net_dims):
        """
        Linear module for 1st output, linear module for 2nd output, 2nd output is exponentiated.
        :param net_dims: list [input_dim, ..., output]
        """
        super(Linear_Normal, self).__init__()
        self.linear_module_mu = nn.Linear(net_dims[0], net_dims[-1])
        self.linear_module_logsigma = nn.Linear(net_dims[0], net_dims[-1])
        self.net_dims = net_dims

    def forward(self, x):
        return self.linear_module_mu(x), torch.exp(self.linear_module_logsigma(x))


class Linear_Normal_Constant_Sigma(nn.Module):
    def __init__(self, net_dims, init_logsigma, fix_logsigma):
        """
        Linear module for 1st output, scalar as expanded vector for 2nd output, 2nd output is exponentiated.
        :param net_dims: list [input_dim, ..., output]
        :param init_logsigma: initialized value for logsigma
        :param fix_logsigma: whether the 2nd output is fixed or a parameter
        """
        super(Linear_Normal_Constant_Sigma, self).__init__()
        self.linear_module_mu = nn.Linear(net_dims[0], net_dims[-1])
        if fix_logsigma:
            self.register_buffer("logsigma", torch.tensor([init_logsigma], dtype=torch.float))
        else:
            self.logsigma = nn.Parameter(torch.tensor([init_logsigma], dtype=torch.float))
        self.net_dims = net_dims

    def forward(self, x):
        return self.linear_module_mu(x), torch.exp(self.logsigma.expand(self.net_dims[-1]))


class Linear_Normal_Constant_Diagonal(nn.Module):
    def __init__(self, net_dims, init_logdiagonal, fix_logdiagonal):
        """
        Linear module for 1st output, vector for 2nd output, 2nd output is exponentiated.
        :param net_dims: list [input_dim, ..., output]
        :param init_logdiagonal: initialized scalar value for logdiagonal
        :param fix_logdiagonal: whether the 2nd output is fixed or a parameter
        """
        super(Linear_Normal_Constant_Diagonal, self).__init__()
        self.linear_module_mu = nn.Linear(net_dims[0], net_dims[-1])
        if fix_logdiagonal:
            self.register_buffer("logdiagonal", init_logdiagonal * torch.ones(net_dims[-1]))
        else:
            self.logdiagonal = nn.Parameter(init_logdiagonal * torch.ones(net_dims[-1]))
        self.net_dims = net_dims

    def forward(self, x):
        return self.linear_module_mu(x), torch.exp(self.logdiagonal)


class Constant_Normal_Constant_Sigma(nn.Module):
    def __init__(self, net_dims, init_mu, init_logsigma, fix_mu, fix_logsigma):
        """
        Vector for 1st output, scalar as expanded vector for 2nd output, 2nd output is exponentiated.
        :param net_dims: list [..., output]
        :param init_mu: initialized scalar value for mu (repeated for each dimension)
        :param init_logsigma: initialized value for logsigma
        :param fix_mu: whether the 1st output is fixed or parameters
        :param fix_logsigma: whether the 2nd output is fixed or a parameter
        """
        super(Constant_Normal_Constant_Sigma, self).__init__()
        if fix_mu:
            self.register_buffer("mu", init_mu * torch.ones(net_dims[-1]))
        else:
            self.mu = nn.Parameter(init_mu * torch.ones(net_dims[-1]))
        if fix_logsigma:
            self.register_buffer("logsigma", torch.tensor([init_logsigma], dtype=torch.float))
        else:
            self.logsigma = nn.Parameter(torch.tensor([init_logsigma], dtype=torch.float))
        self.net_dims = net_dims

    def forward(self, *args):
        return self.mu, torch.exp(self.logsigma.expand(self.net_dims[-1]))


class Identity_Normal_Constant_Sigma(nn.Module):
    def __init__(self, net_dims, init_logsigma, fix_logsigma):
        """
        Identity module for 1st output, scalar as expanded vector for 2nd output, 2nd output is exponentiated.
        :param net_dims: list [..., output]
        :param init_logsigma: initialized value for logsigma
        :param fix_logsigma: whether the 2nd output is fixed or a parameter
        """
        super(Identity_Normal_Constant_Sigma, self).__init__()
        if fix_logsigma:
            self.register_buffer("logsigma", torch.tensor([init_logsigma], dtype=torch.float))
        else:
            self.logsigma = nn.Parameter(torch.tensor([init_logsigma], dtype=torch.float))
        self.net_dims = net_dims

    def forward(self, x):
        return x, torch.exp(self.logsigma.expand(self.net_dims[-1]))
