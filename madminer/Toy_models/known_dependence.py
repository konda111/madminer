import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def fun(x):
    return (10+10*x+10*x**2)
def matrix_system(x):
    x0 = x[0]
    x1 = x[1]
    x2 = x[2]
    return [[1,x0,x0**2],[1,x1,x1**2],[1,x2,x2**2]]



basis_trial = np.random.uniform(-20,20,3)

def minimal(basis_trial):
    mat = matrix_system(basis_trial)
    inverse = np.linalg.inv(mat)
    sigmas = []
    for row in np.abs(inverse):
        sigmas.append(np.sqrt(np.sum(row*fun(basis_trial)**2)))
    return np.sum(sigmas)

print(minimize(minimal,basis_trial,method="Nelder-Mead"))







    




