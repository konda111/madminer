import logging

import numpy as np
import torch
import math
from torch.nn import BCELoss
from torch.nn import MSELoss

logger = logging.getLogger(__name__)

# RBF kernel with median estimator
def kernel(x, y):
    channels = len(x)
    dim_score = x.shape[-1]
    dnorm2 = (x.reshape(channels,1,-1, dim_score) - y.reshape(1,channels,-1, dim_score)).square().sum(dim=2)
    sigma = torch.quantile(dnorm2.detach(), 0.5) / (2 * math.log(channels + 1))
    return torch.exp(- dnorm2 / (2*sigma))

def ratio_mse_num(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, log_r_clip=10.0):
    r_true = torch.clamp(r_true, np.exp(-log_r_clip), np.exp(log_r_clip))
    log_r_hat = torch.clamp(log_r_hat, -log_r_clip, log_r_clip)

    inverse_r_hat = torch.exp(-log_r_hat)
    return MSELoss()((1.0 - y_true) * inverse_r_hat, (1.0 - y_true) * (1.0 / r_true))


def ratio_mse_den(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, log_r_clip=10.0):
    r_true = torch.clamp(r_true, np.exp(-log_r_clip), np.exp(log_r_clip))
    log_r_hat = torch.clamp(log_r_hat, -log_r_clip, log_r_clip)

    r_hat = torch.exp(log_r_hat)
    return MSELoss()(y_true * r_hat, y_true * r_true)


def ratio_mse(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, log_r_clip=10.0):
    return ratio_mse_num(
        s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, log_r_clip
    ) + ratio_mse_den(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, log_r_clip)


def ratio_score_mse_num(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true):
    return MSELoss()((1.0 - y_true) * t0_hat, (1.0 - y_true) * t0_true)


def ratio_score_mse_den(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true):
    return MSELoss()(y_true * t1_hat, y_true * t1_true)


def ratio_score_mse(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true):
    return ratio_score_mse_num(
        s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true
    ) + ratio_score_mse_den(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true)


def ratio_xe(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true):
    s_hat = 1.0 / (1.0 + torch.exp(log_r_hat))

    return BCELoss()(s_hat, y_true)


def ratio_augmented_xe(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, reduction='mean'):
    s_hat = 1.0 / (1.0 + torch.exp(log_r_hat))
    s_true = 1.0 / (1.0 + r_true)
    return BCELoss(reduction=reduction)(s_hat, s_true)

def repulsive_ratio_augmented_xe(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, data_len):
    # first compute ratio_augmented_xe loss
    losses = ratio_augmented_xe(s_hat, log_r_hat, t0_hat, t1_hat, y_true, r_true, t0_true, t1_true, reduction='none')
    n_channels = losses.shape[0]
    # repulsive ensemble loss
    k = kernel(losses, losses.detach())
    losses_mean, losses_std = losses.mean(dim=1), losses.std(dim=1)
    loss = torch.sum(losses_mean + (k.sum(dim=1) / k.detach().sum(dim=1) - 1) / data_len, dim=0) # loss shape: (n_parameters)
    return loss.sum()/n_channels

def local_score_mse(t_hat, t_true):
    return MSELoss()(t_hat, t_true)


def heteroskedastic_loss(outputs, t_true):
    mus = outputs[:, 0]
    logsigma2s = outputs[:, 1]
    out = torch.pow(mus - t_true.reshape(-1), 2)/(2 * logsigma2s.exp()) + 1/2. * logsigma2s
    return torch.mean(out)

def repulsive_ensemble_loss(outputs, t_true, data_len):
    # first compute heteroskedastic regression loss
    mus = outputs[:, :, :, 0]
    logsigma2s = outputs[:, :, :, 1]
    reg = torch.pow(mus - t_true.reshape(mus.shape), 2)/(2 * logsigma2s.exp()) + 1/2. * logsigma2s
    # repulsive ensemble loss
    k = kernel(reg, reg.detach())
    reg_mean, reg_std = reg.mean(dim=1), reg.std(dim=1)
    loss = torch.sum(reg_mean + (k.sum(dim=1) / k.detach().sum(dim=1) - 1) / data_len, dim=0) # loss shape: (n_parameters)
    return loss.sum()

def repulsive_ensemble_mse_loss(outputs, t_true, data_len):
    mus = outputs[:, :, :, 0]
    reg = torch.pow(mus - t_true.reshape(mus.shape), 2)
    # repulsive ensemble loss
    k = kernel(reg, reg.detach())
    n_channels = reg.shape[0]
    reg_mean, reg_std = reg.mean(dim=1), reg.std(dim=1)
    loss = torch.sum(reg_mean + (k.sum(dim=1) / k.detach().sum(dim=1) - 1) / data_len, dim=0) # loss shape: (n_parameters)
    return loss.sum()/n_channels

def bayesian_loss(model, outputs, t_true):
    nl = model.neg_log_gauss(outputs, t_true.reshape(-1))
    kl = model.KL(len(outputs))
    return nl + kl

def bayesian_mse_loss(model, outputs, t_true):
    t_pred = outputs[:, 0]
    return MSELoss()(t_pred, t_true.flatten())

def local_score_mse_weighted(t_hat, t_true, weights):
    return (weights * (t_hat - t_true) ** 2).mean()


def flow_nll(log_p_pred, t_pred, t_true):
    return -torch.mean(log_p_pred)


def flow_score_mse(log_p_pred, t_pred, t_true):
    return MSELoss()(t_pred, t_true)
