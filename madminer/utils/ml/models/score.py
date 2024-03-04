import logging

import torch
import torch.nn as nn
import numpy as np

from torch.autograd import grad
from madminer.utils.ml.utils import get_activation_function

logger = logging.getLogger(__name__)


class DenseLocalScoreModel(nn.Module):
    """Module that implements local score estimators for methods like SALLY and SALLINO, or the calculation
    of Fisher information matrices."""

    def __init__(self, n_observables, n_parameters, n_hidden, activation="tanh", dropout_prob=0.0):
        super().__init__()

        # Save input
        self.n_hidden = n_hidden
        self.activation = get_activation_function(activation)
        self.dropout_prob = dropout_prob

        # Build network
        self.layers = nn.ModuleList()
        n_last = n_observables

        # Hidden layers
        for n_hidden_units in n_hidden:
            if self.dropout_prob > 1.0e-9:
                self.layers.append(nn.Dropout(self.dropout_prob))
            self.layers.append(nn.Linear(n_last, n_hidden_units))
            n_last = n_hidden_units

        # Log r layer
        if self.dropout_prob > 1.0e-9:
            self.layers.append(nn.Dropout(self.dropout_prob))
        self.layers.append(nn.Linear(n_last, n_parameters))

    def forward(self, x, return_grad_x=False):
        # Track gradient wrt x
        if return_grad_x and not x.requires_grad:
            x.requires_grad = True

        # Forward pass
        t_hat = x

        for i, layer in enumerate(self.layers):
            if i > 0:
                t_hat = self.activation(t_hat)
            t_hat = layer(t_hat)

        # Calculate gradient
        if return_grad_x:
            x_gradient = grad(
                t_hat,
                x,
                grad_outputs=torch.ones_like(t_hat.data),
                only_inputs=True,
                create_graph=True,
            )[0]

            return t_hat, x_gradient

        return t_hat

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)

        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(*args, **kwargs)

        return self
    
class HeteroSkedasticDenseLocalScoreModel(DenseLocalScoreModel):
    """Module that implements local score estimators for methods like SALLY and SALLINO, or the calculation
    of Fisher information matrices. Modified version of DenseLocalScoreModel using heteroskedastic loss."""

    def __init__(self, n_observables, n_parameters, n_hidden, activation="tanh", dropout_prob=0.0):
        super().__init__(
            n_observables,
            n_parameters,
            n_hidden,
            activation=activation,
            dropout_prob=dropout_prob
        )

        # Save input
        self.n_hidden = n_hidden
        self.activation = get_activation_function(activation)
        self.dropout_prob = dropout_prob

        # Build network
        self.layers = nn.ModuleList()
        n_last = n_observables

        # Hidden layers
        for n_hidden_units in n_hidden:
            if self.dropout_prob > 1.0e-9:
                self.layers.append(nn.Dropout(self.dropout_prob))
            self.layers.append(nn.Linear(n_last, n_hidden_units))
            n_last = n_hidden_units

        # Log r layer
        if self.dropout_prob > 1.0e-9:
            self.layers.append(nn.Dropout(self.dropout_prob))
        # output is a vector of size 2*n_parameters
        # saving the logr mean and width for each parameter sequentially (mu_1, sigma_1, mu_2, sigma_2, ...)
        self.layers.append(nn.Linear(n_last, 2*n_parameters))


class VBLinear(nn.Module):
    # VB -> Variational Bayes
    
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
    
    
class BayesianDenseLocalScoreModel(DenseLocalScoreModel):
    """Module that implements local score estimators for methods like SALLY and SALLINO, or the calculation
    of Fisher information matrices."""

    def __init__(self, n_observables, n_parameters, n_hidden, activation="tanh", dropout_prob=0.0):
        
        super().__init__(
            n_observables,
            n_parameters,
            n_hidden,
            activation=activation,
            dropout_prob=dropout_prob
        )

        # Save input
        self.n_parameters = n_parameters
        self.n_hidden = n_hidden
        self.activation = get_activation_function(activation)
        self.dropout_prob = dropout_prob

        # Build network
        self.layers = nn.ModuleList()
        self.vb_layers = []
        n_last = n_observables
        
        # Hidden layers
        for n_hidden_units in n_hidden:
            if self.dropout_prob > 1.0e-9:
                self.layers.append(nn.Dropout(self.dropout_prob))
            vb_layer = VBLinear(n_last, n_hidden_units)
            self.vb_layers.append(vb_layer)
            self.layers.append(vb_layer)
            self.layers.append(nn.ReLU())
            n_last = n_hidden_units

        # Log r layer
        if self.dropout_prob > 1.0e-9:
            self.layers.append(nn.Dropout(self.dropout_prob))
        # output is a vector of size 2*n_parameters
        # saving the logr mean and width for each parameter sequentially (mu_1, sigma_1, mu_2, sigma_2, ...)
        vb_layer = VBLinear(n_last, 2*n_parameters)
        self.vb_layers.append(vb_layer)
        self.layers.append(vb_layer)
    
    def KL(self, training_size):
        kl = 0
        for vb_layer in self.vb_layers:
            kl += vb_layer.KL()
        return kl / training_size
    
    def neg_log_gauss(self, outputs, targets):
        # output has form [[mu1, logsigmaSq1], [mu2, logsigmaSq2], ..., [mu_n, logsigmaSq_n]]
        mus = outputs[:, 0]
        logsigma2s = outputs[:, 1]
        out = torch.pow(mus - targets, 2)/(2 * logsigma2s.exp()) + 1./2. * logsigma2s
        return torch.mean(out)