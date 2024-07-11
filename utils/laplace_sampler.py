import torch
import torch.nn as nn
import torch.distributions as tdist


class GetLaplaceData():
    def __init__(self, mean=[0.0], lamb=[1.0], device='cpu'):
        super(GetLaplaceData, self).__init__()
        # loc: mean, scale: b (look at wikipedia for b) where variance = 2*b^2
        self.laplace = tdist.Laplace(loc=torch.tensor(mean, device=device), scale=torch.tensor(lamb, device=device))

    def getdata(self, tsize=torch.Size((1, 1, 1, 1))):
        # NOTE: each channel can have different mean, loc. Initiliaze tdist above accordingly
        assert tsize[1] == 1  # channel dim of tsize should be 0. Will have in channel dimension diffedrent distributns
        ss = self.laplace.sample(tsize)
        return ss.permute([0, 4, 2, 3, 1]).squeeze_(dim=4)

    def get_stddev(self):
        return self.laplace.stddev
