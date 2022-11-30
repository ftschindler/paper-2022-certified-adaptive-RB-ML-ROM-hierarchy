# +
import torch



# Implementation of a few kernels
from abc import ABC, abstractmethod




class Kernel(ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def diagonal(self, X):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass


class RBF(Kernel):
    @abstractmethod
    def __init__(self):
        super(RBF, self).__init__()

    def eval(self, x, y):
        return self.rbf(self.ep, torch.cdist(x, y))

    def diagonal(self, X):
        return torch.ones(X.shape[0], 1) * self.rbf(self.ep, torch.tensor(0.0))

    def __str__(self):
        return self.name + ' [gamma = %2.2e]' % self.ep

    def set_params(self, par):
        self.ep = par


class Gaussian(RBF):
    def __init__(self, ep=1):
        self.ep = ep
        self.name = 'gauss'
        self.rbf = lambda ep, r: torch.exp(-(ep * r) ** 2)

class Matern(RBF):
    def __init__(self, ep=1):
        self.ep = ep
        self.name = 'matern'
        self.rbf = lambda ep, r: torch.exp(-ep * r)

class Wendland_order_0(RBF):
    def __init__(self, ep=1):
        self.ep = ep
        # self.rbf = lambda ep, r: torch.clamp(1 - ep*r, min=0)
        self.rbf = lambda ep, r: torch.nn.functional.relu(1- ep * r)
        self.name = 'Wendland order 0'

