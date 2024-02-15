import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.optimize import minimize


N_wilson = 1
N_obs = 1

mus = np.asarray([np.linspace(-2+j/3,2-j/3,N_obs) for j in range(N_wilson+1)])
sigmas = np.identity(mus.shape[1])

#This defines some standard x_range. At worst roughly we are talking about [-15,15] per dimension (-10-5*sigmas,10+5*sigmas).
#We restric the simulation to this box

#In future to compare the various fixed vs non fixed theta I need to select some box length for theta. for now choose theta_max = 4, theta_min = -4

L_box_random_theta = []
L_box_fixed_theta = []
for j in range(N_wilson):
    L_box_random_theta.append(4)
for j in range(N_obs):
    L_box_random_theta.append(np.max(np.abs(mus)))
    L_box_fixed_theta.append(np.max(np.abs(mus)))

L_box_random_theta = np.asarray(L_box_random_theta)
L_box_fixed_theta = np.asarray(L_box_fixed_theta)
# I also need to define some maximum for the integration. In the fixed theta case this can be done theta by theta, in the random theta one I select given
# dimension of the theta box etc something which makes sense. 

max_rnd_theta = N_wilson*4**2
volume_rnd_theta = max_rnd_theta*np.prod(2*L_box_random_theta)
def dsigma(ipt):
    if ipt.ndim == 1:
        th = ipt[0:N_wilson]
        th = np.insert(th,0,1)
        xs = ipt[N_wilson:N_wilson+N_obs]
        funs = np.asarray([multivariate_normal.pdf(xs,mean=m,cov=sigmas) for m in mus])
        return ((th*funs).sum())**2
    if ipt.ndim == 2:
        th = ipt[:,0:N_wilson]
        th = np.column_stack((np.ones(ipt.shape[0]),th))
        xs = ipt[:,N_wilson:N_wilson+N_obs]
        funs = np.asarray([multivariate_normal.pdf(xs,mean=m,cov=sigmas) for m in mus])
        return ((th*funs.T).sum(-1))**2

def dsigma_0(ipt_x_only):
    th = np.zeros(N_wilson)
    th = np.insert(th,0,1)
    xs = ipt_x_only
    funs = np.asarray([multivariate_normal.pdf(xs,mean=m,cov=sigmas) for m in mus])
    return ((th*funs).sum())**2
    
def simulation(N_reps,random_theta = True, theta = None):
    if random_theta:
        n_effective = 0
        n_tot = 0
        while n_effective < N_reps:
            alpha = (np.random.random((100*N_reps,N_wilson+N_obs))-0.5)*2
            alpha = alpha*L_box_random_theta # this is the set of trial points
            function_val = dsigma(alpha)
            ys = np.random.uniform(0,max_rnd_theta,100*N_reps)
            mask = function_val > ys # keep these events
            n_effective += np.sum(mask)
            try:
                final_events = np.vstack((final_events,alpha[mask]))
            except:
                final_events = alpha[mask]
            n_tot += 100*N_reps
            print(n_effective)
        final_events = final_events[0:N_reps]
        sigma = n_effective/n_tot*volume_rnd_theta
        return final_events, sigma
    if not random_theta:
        n_effective = 0
        n_tot = 0
        while n_effective < N_reps:
            alpha = (np.random.random((100*N_reps,N_obs))-0.5)*2
            alpha = alpha*L_box_fixed_theta
            alpha = np.column_stack((np.tile(theta, (100*N_reps, 1)),alpha))
            function_val = dsigma(alpha)
            ys = np.random.uniform(0,max_rnd_theta,100*N_reps)
            mask = function_val > ys # keep these events
            n_effective += np.sum(mask)
            print(n_effective)
            try:
                final_events = np.vstack((final_events,alpha[mask]))
            except:
                final_events = alpha[mask]
            n_tot += 100*N_reps
        final_events = final_events[0:N_reps]
        sigma = n_effective/n_tot*volume_rnd_theta
        return final_events, sigma


#I have the functions which generate events, both for fixed theta and for random theta
#In principle I have everything I need to train CARL on the generated data. I still need to build the minimizer for the fixed benches stuff

import os
import logging
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

from madminer.ml import ParameterizedRatioEstimator
if not os.path.exists("data"):
    os.makedirs("data")

# MadMiner output
logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.INFO,
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)

sim_0 = simulation(100)
print("random sim done")
x_from_theta0 = sim_0[0][:,N_wilson:N_wilson+N_obs]
theta0 = sim_0[0][:,0:N_wilson]
x_from_theta1 = simulation(100,False,np.zeros(N_wilson))[0][:,N_wilson:N_wilson+N_obs]

x_train = np.hstack((x_from_theta0, x_from_theta1)).reshape(-1, N_obs)
y_train = np.hstack((np.zeros(x_from_theta0.shape[0]), np.zeros(x_from_theta1.shape[0]))).reshape(-1, 1)
theta0_train = np.hstack((theta0, theta0)).reshape(-1, N_wilson)
np.save("data/theta0_train.npy", theta0_train)
np.save("data/x_train.npy", x_train)
np.save("data/y_train.npy", y_train)
#import ipdb; ipdb.set_trace()
carl = ParameterizedRatioEstimator(n_hidden=(20, 20))

carl.train(
    method="carl",
    x="data/x_train.npy",
    y="data/y_train.npy",
    theta="data/theta0_train.npy",
    n_epochs=20,
)

carl.save("models/carl")

#Compute sigmas coefficients; structure is s0+s1th+...; Find polynomial coming from square from matrix multiplication thing
def coefficients_vector(theta):
    th = theta
    th = np.insert(th,0,1)
    mat = th[:,np.newaxis]@th[np.newaxis,:]
    mat[np.triu_indices(N_wilson+1, k = 1)]*=2
    return mat[np.triu_indices(N_wilson+1)]
sigma_basis = []
for j in range(int((N_wilson**2+2+3*N_wilson)/2)):
    sigma_basis.append(np.random.uniform(-5,5,N_wilson))

# Check that it is invertible!!
ok = False
while ok == False:
    matrix_system = []
    for elem in sigma_basis:
        matrix_system.append(coefficients_vector(elem))
    print((np.shape(matrix_system)))
    try: 
        inverse = np.linalg.inv(matrix_system)
        ok = True
    except:
        print("Failed inversion")
        ok = False
        sigma_basis = []
        for j in range(int((N_wilson**2+2+2*N_wilson)/2)):
            sigma_basis.append(np.random.uniform(-5,5,N_wilson))

sigma_values = []
for elem in sigma_basis:
    sigma_values.append(simulation(100,random_theta = False, theta = elem)[1])

## Infer coefficients
coeffs = inverse@sigma_values
## Now that I have "rough" estimate of coefficients I can lay down the optimization algorithm for the ratio morphing itself.
# TODO: This is actually too imprecise sometimes. For now or at least for this toy model beyond 2 N_wilson this does 
# not work well
normed_coeffs = coeffs/coeffs[0]
#coeffs[0] is == sigma(0)
def fun_to_minim(basis):
    basis = np.reshape(basis,(int((N_wilson**2+2+3*N_wilson)/2),N_wilson))


















    