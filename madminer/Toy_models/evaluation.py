import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import os
import logging
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

from madminer.ml import MorphParameterizedRatioEstimator
if not os.path.exists("data"):
    os.makedirs("data")

# MadMiner output
logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.INFO,
)


N_wilson = 1
N_obs = 1
ratio_basis = np.load("bench_basis.npy")
def coefficients_vector(theta):
    th = theta
    th = np.insert(th,0,1)
    mat = th[:,np.newaxis]@th[np.newaxis,:]
    mat[np.triu_indices(N_wilson+1, k = 1)]*=2
    return mat[np.triu_indices(N_wilson+1)]

coefficients = np.load("sigma_coefficients.npy")

def sigma_at_theta(theta):
    return coefficients@coefficients_vector(theta)

model_list = []

for j in range(np.shape(ratio_basis)[0]):
    carl = MorphParameterizedRatioEstimator(n_hidden=(20, 20))
    carl.load(f"models/carl_{j}")
    model_list.append(carl)


## I have an array of samples and of sigmas

#I have the functions which generate events, both for fixed theta and for random theta
#In principle I have everything I need to train CARL on the generated data. I still need to build the minimizer for the fixed benches stuff

def ratio_at_theta(benchmark_ratios,theta, benchmark_basis, sigmas_at_benchmark):
    matrix = []
    for j,elem in enumerate(benchmark_basis):
        matrix.append(coefficients_vector(elem)/sigmas_at_benchmark[j])
    inv = np.linalg.inv(matrix)
    return (coefficients_vector(theta)/sigma_at_theta(theta))@inv@benchmark_ratios
