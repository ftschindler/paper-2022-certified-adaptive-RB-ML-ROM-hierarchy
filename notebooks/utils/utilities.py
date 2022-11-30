# +
import math
import torch

from utils import kernels



class ActivFunc(torch.nn.Module):
    '''
    Implementation of the single-dimensional kernel layers of the SDKN, which can be viewed
    as optimizable activation function layers.
    '''

    def __init__(self, in_features, nr_centers, kernel=None):
        super().__init__()

        # General stuff
        self.in_features = in_features
        self.nr_centers = nr_centers
        self.nr_centers_id = nr_centers  # number of centers + maybe additional dimension for identity

        # Define kernel if not given
        if kernel is None:
            self.kernel = kernels.Wendland_order_0(ep=1)
        else:
            self.kernel = kernel

        # Weight parameters
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_features, self.nr_centers_id))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data += .2 * torch.ones(self.weight.data.shape)  # provide some positive mean

    def forward(self, x, centers):
        cx = torch.cat((centers, x), 0)

        dist_matrix = torch.abs(torch.unsqueeze(cx, 2) - centers.t().view(1, centers.shape[1], self.nr_centers))
        kernel_matrix = self.kernel.rbf(self.kernel.ep, dist_matrix)
        cx = torch.sum(kernel_matrix * self.weight, dim=2)

        centers = cx[:self.nr_centers, :]
        x = cx[self.nr_centers:, :]

        return x, centers

