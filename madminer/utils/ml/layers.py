import logging

import torch
import torch.nn as nn
import numpy as np
import math

from torch.autograd import grad
from madminer.utils.ml.utils import get_activation_function

logger = logging.getLogger(__name__)


class VBLinear(nn.Module):
    """
    Bayesian linear layer with variational inference
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor
    
    def __init__( self, in_features, out_features ):
        super(VBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.resample = True
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.logsig2_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.random = torch.randn_like(self.logsig2_w)
        self.reset_parameters()
        
    def forward(self, input):
        if self.resample:
            self.random = torch.randn_like(self.logsig2_w)
        s2_w = self.logsig2_w.exp()
        weight = self.mu_w + s2_w.sqrt() * self.random
        return nn.functional.linear(input, weight, self.bias) #+ 1e-8
    
    def reset_parameters( self ):
        stdv = 1. / np.sqrt( self.mu_w.size(1) )
        self.mu_w.data.normal_( 0, stdv )
        self.logsig2_w.data.zero_().normal_( -9, 0.001 )
        self.bias.data.zero_()
        
    def KL( self, loguniform=False ):
        kl = 0.5 * ( self.mu_w.pow(2) + self.logsig2_w.exp() - self.logsig2_w - 1 ).sum()
        return kl
    

class StackedLinear(nn.Module):
    """
    Efficient implementation of linear layers for ensembles of networks
    """
    def __init__(self, in_features, out_features, channels):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channels = channels
        self.weight = nn.Parameter(torch.empty((channels, out_features, in_features)))
        self.bias = nn.Parameter(torch.empty((channels, out_features)))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.channels):
            torch.nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[i])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, input):
        return torch.baddbmm(self.bias[:,None,:], input, self.weight.transpose(1,2))