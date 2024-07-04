import logging

import numpy as np
import torch
from torch import tensor

from madminer.utils.ml.models.ratio import DenseSingleParameterizedRatioModel
from madminer.utils.ml.models.ratio import RepulsiveEnsembleDenseSingleParameterizedRatioModel
from madminer.utils.ml.models.ratio import BayesianDenseSingleParameterizedRatioModel
from madminer.utils.ml.models.ratio import DenseDoublyParameterizedRatioModel

logger = logging.getLogger(__name__)


def evaluate_flow_model(model, thetas=None, xs=None, evaluate_score=False, run_on_gpu=True, double_precision=False):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Balance theta0 and theta1
    n_thetas = len(thetas)

    # Prepare data
    n_xs = len(xs)
    thetas = torch.stack([tensor(thetas[i % n_thetas], requires_grad=True) for i in range(n_xs)])
    xs = torch.stack([tensor(i) for i in xs])

    model = model.to(device, dtype)
    thetas = thetas.to(device, dtype)
    xs = xs.to(device, dtype)

    # Evaluate estimator with score:
    if evaluate_score:
        model.eval()

        _, log_p_hat, t_hat = model.log_likelihood_and_score(thetas, xs)

        # Copy back tensors to CPU
        if run_on_gpu:
            log_p_hat = log_p_hat.cpu()
            t_hat = t_hat.cpu()

        log_p_hat = log_p_hat.detach().numpy().flatten()
        t_hat = t_hat.detach().numpy().flatten()

    # Evaluate estimator without score:
    else:
        with torch.no_grad():
            model.eval()

            _, log_p_hat = model.log_likelihood(thetas, xs)

            # Copy back tensors to CPU
            if run_on_gpu:
                log_p_hat = log_p_hat.cpu()

            log_p_hat = log_p_hat.detach().numpy().flatten()
            t_hat = None

    return log_p_hat, t_hat


def evaluate_ratio_model(
    model,
    method_type=None,
    theta0s=None,
    theta1s=None,
    xs=None,
    evaluate_score=False,
    run_on_gpu=True,
    double_precision=False,
    return_grad_x=False,
    n_eval=100
):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Figure out method type
    if method_type is None:
        if isinstance(model, RepulsiveEnsembleDenseSingleParameterizedRatioModel):
            method_type = "repulsive_parameterized"
        elif isinstance(model, BayesianDenseSingleParameterizedRatioModel):
            method_type = "bayesian_parameterized"
        elif isinstance(model, DenseSingleParameterizedRatioModel):
            method_type = "parameterized"
        elif isinstance(model, DenseDoublyParameterizedRatioModel):
            method_type = "doubly_parameterized"
        else:
            raise RuntimeError("Cannot infer method type automatically")

    # Balance theta0 and theta1
    if theta1s is None:
        n_thetas = len(theta0s)
    else:
        n_thetas = max(len(theta0s), len(theta1s))
        if len(theta0s) > len(theta1s):
            theta1s = np.array([theta1s[i % len(theta1s)] for i in range(len(theta0s))])
        elif len(theta0s) < len(theta1s):
            theta0s = np.array([theta0s[i % len(theta0s)] for i in range(len(theta1s))])

    # Prepare data
    n_xs = len(xs)
    theta0s = torch.stack([tensor(theta0s[i % n_thetas], requires_grad=evaluate_score) for i in range(n_xs)])
    if theta1s is not None:
        theta1s = torch.stack([tensor(theta1s[i % n_thetas], requires_grad=evaluate_score) for i in range(n_xs)])
    xs = torch.stack([tensor(i) for i in xs])
    
    if method_type == "repulsive_parameterized":
        xs = xs[None,:].expand(model.n_channels,-1,-1)
        theta0s = theta0s[None,:].expand(model.n_channels,-1,-1)

    model = model.to(device, dtype)
    theta0s = theta0s.to(device, dtype)
    if theta1s is not None:
        theta1s = theta1s.to(device, dtype)
    xs = xs.to(device, dtype)

    # Evaluate ratio estimator with score or x gradients:
    if evaluate_score or return_grad_x:
        model.eval()

        if method_type == "parameterized_ratio":
            if return_grad_x:
                s_hat, log_r_hat, t_hat0, x_gradients = model(
                    theta0s,
                    xs,
                    return_grad_x=True,
                    track_score=evaluate_score,
                    create_gradient_graph=False,
                )
            else:
                s_hat, log_r_hat, t_hat0 = model(
                    theta0s,
                    xs,
                    track_score=evaluate_score,
                    create_gradient_graph=False,
                )
                x_gradients = None
            t_hat1 = None

        elif method_type == "double_parameterized_ratio":
            if return_grad_x:
                s_hat, log_r_hat, t_hat0, t_hat1, x_gradients = model(
                    theta0s,
                    theta1s,
                    xs,
                    return_grad_x=True,
                    track_score=evaluate_score,
                    create_gradient_graph=False,
                )
            else:
                s_hat, log_r_hat, t_hat0, t_hat1 = model(
                    theta0s,
                    theta1s,
                    xs,
                    track_score=evaluate_score,
                    create_gradient_graph=False,
                )
                x_gradients = None
        else:
            raise ValueError("Unknown method type %s", method_type)

        # Copy back tensors to CPU
        if run_on_gpu:
            s_hat = s_hat.cpu()
            log_r_hat = log_r_hat.cpu()
            if t_hat0 is not None:
                t_hat0 = t_hat0.cpu()
            if t_hat1 is not None:
                t_hat1 = t_hat1.cpu()

        # Get data and return
        s_hat = s_hat.detach().numpy().flatten()
        log_r_hat = log_r_hat.detach().numpy().flatten()
        if t_hat0 is not None:
            t_hat0 = t_hat0.detach().numpy()
        if t_hat1 is not None:
            t_hat1 = t_hat1.detach().numpy()

    # Evaluate ratio estimator without score:
    else:
        with torch.no_grad():
            model.eval()

            if method_type == "parameterized_ratio":
                s_hat, log_r_hat, _ = model(theta0s, xs, track_score=False, create_gradient_graph=False)
            elif method_type == "double_parameterized_ratio":
                s_hat, log_r_hat, _, _ = model(theta0s, theta1s, xs, track_score=False, create_gradient_graph=False)
            elif method_type == "repulsive_parameterized":
                s_hat, log_r_hat, _ = model(theta0s, xs, track_score=False, create_gradient_graph=False)
            elif method_type == "bayesian_parameterized":
                s_hat, log_r_hat = [], []
                for _ in range(n_eval):
                    s_hat_one, log_r_hat_one, _ = model(theta0s, xs, track_score=False, create_gradient_graph=False)
                    # print(f"log_r_hat_one neg: {log_r_hat_one.shape, log_r_hat_one[log_r_hat_one < 0].shape}")
                    s_hat.append(s_hat_one.detach().numpy())
                    log_r_hat_one = log_r_hat_one[:, 0] # only use mu 
                    # print(log_r_hat_one)
                    # log_r_hat_one = torch.log(log_r_hat_one)
                    log_r_hat.append(log_r_hat_one.detach().numpy())
                s_hat = np.array(s_hat).T
                log_r_hat = np.array(log_r_hat)
                log_r_hat_mean = log_r_hat.mean(axis=0).flatten()
                log_r_hat_std = log_r_hat.std(axis=0).flatten()
            else:
                raise ValueError("Unknown method type %s", method_type)

            # Copy back tensors to CPU
            if run_on_gpu:
                s_hat = s_hat.cpu()
                log_r_hat = log_r_hat.cpu()

            # Get data and return
            if method_type != "bayesian_parameterized":
                s_hat = s_hat.detach().numpy().flatten()
            if method_type == "repulsive_parameterized":
                log_r_hat_mean = log_r_hat.mean(dim=0).flatten().numpy()
                log_r_hat_std = log_r_hat.std(dim=0).flatten().numpy()
            elif method_type == "bayesian_parameterized":
                pass
            else:
                log_r_hat = log_r_hat.detach().numpy().flatten()
            t_hat0, t_hat1 = None, None
    if method_type in ["repulsive_parameterized", "bayesian_parameterized"]:
        if return_grad_x:
            return s_hat, log_r_hat, t_hat0, t_hat1, x_gradients, log_r_hat_std
        return s_hat, log_r_hat_mean, t_hat0, t_hat1, log_r_hat_std
    else:
        if return_grad_x:
            return s_hat, log_r_hat, t_hat0, t_hat1, x_gradients
        return s_hat, log_r_hat, t_hat0, t_hat1


def evaluate_local_score_model(model, xs=None, run_on_gpu=True, double_precision=False, return_grad_x=False):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Prepare data
    xs = torch.stack([tensor(i) for i in xs])

    model = model.to(device, dtype)
    xs = xs.to(device, dtype)

    # Evaluate networks
    if return_grad_x:
        model.eval()
        t_hat, x_gradients = model(xs, return_grad_x=True)
    else:
        with torch.no_grad():
            model.eval()
            t_hat = model(xs)
        x_gradients = None

    # Copy back tensors to CPU
    if run_on_gpu:
        t_hat = t_hat.cpu()
        if x_gradients is not None:
            x_gradients = x_gradients.cpu()

    # Get data and return
    t_hat = t_hat.detach().numpy()

    if return_grad_x:
        x_gradients = x_gradients.detach().numpy()
        return t_hat, x_gradients

    return t_hat

def evaluate_unc_local_score_model(
    model, 
    xs=None, 
    run_on_gpu=True, 
    double_precision=False, 
    return_grad_x=False,
    n_eval=100
):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Prepare data
    xs = torch.stack([tensor(i) for i in xs])

    model = model.to(device, dtype)
    xs = xs.to(device, dtype)

    # Evaluate networks
    t_hat_list = []
    x_gradients_list = []
    for _ in range(n_eval):
        if return_grad_x:
            model.eval()
            t_hat, x_gradients = model(xs, return_grad_x=True)
        else:
            with torch.no_grad():
                model.eval()
                t_hat = model(xs)
            x_gradients = None

        # Copy back tensors to CPU
        if run_on_gpu:
            t_hat = t_hat.cpu()
            if x_gradients is not None:
                x_gradients = x_gradients.cpu()

        # Get data and return
        t_hat = t_hat.detach().numpy()
        t_hat_mu = t_hat[:, ::2]
        t_hat_sig = np.exp(.5*t_hat[:, 1::2])
        t_hat_list.append([t_hat_mu, t_hat_sig])
        if return_grad_x:
            x_gradients = x_gradients.detach().numpy()
            x_gradients_list.append(x_gradients)
    
    t_hat_list = np.array(t_hat_list)
    x_gradients_list = np.array(x_gradients_list)
    
    t_hat_mu = t_hat_list[:,0].mean(axis=0)
    t_hat_sig = t_hat_list[:,0].std(axis=0)
    t_hat_sig_stoch = np.sqrt(np.mean(t_hat_list[:,1], axis=0))
    t_hat_sig_tot = np.sqrt(t_hat_sig**2 + t_hat_sig_stoch**2)
    if return_grad_x:
        return t_hat_mu, t_hat_sig, t_hat_sig_stoch, t_hat_sig_tot, x_gradients

    return t_hat_mu, t_hat_sig, t_hat_sig_stoch, t_hat_sig_tot

def evaluate_repulsive_ensemble_local_score_model(
    model, 
    xs=None, 
    run_on_gpu=True, 
    double_precision=False, 
    return_grad_x=False, 
    return_individual_contributions=False
):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.backends.mps.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Prepare data
    xs = torch.stack([tensor(i) for i in xs])
    x = xs.clone().detach()
    x = x[None,:].expand(model.n_channels,-1,-1)

    model = model.to(device, dtype)
    x = x.to(device, dtype)

    # Evaluate networks
    if return_grad_x:
        model.eval()
        t_hat, x_gradients = model(x, return_grad_x=True)
    else:
        with torch.no_grad():
            model.eval()
            t_hat = model(x)
        x_gradients = None

    # Copy back tensors to CPU
    if run_on_gpu:
        t_hat = t_hat.cpu()
        if x_gradients is not None:
            x_gradients = x_gradients.cpu()

    # Get data and return
    t_hat = t_hat.detach()
    t_hat = torch.reshape(t_hat, (model.n_channels, xs.shape[0], model.n_parameters, 1)) # bring outputs to shape (n_channels, n_data, n_parameters, 2)
    output = t_hat[:, :, :, 0].numpy()
    
    if return_individual_contributions:
        return output
    
    t_hat_mu = np.mean(output, axis=0)
    m = output - output.sum(0,keepdims=True)/model.n_channels
    t_hat_cov = np.einsum('ijk,ijl->jkl', m, m)/(model.n_channels)
    if return_grad_x:
        x_gradients = x_gradients.detach().numpy()
        return t_hat_mu, t_hat_cov, x_gradients

    return t_hat_mu, t_hat_cov
