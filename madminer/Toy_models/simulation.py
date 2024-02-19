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

from madminer.ml import ParameterizedRatioEstimator, MorphParameterizedRatioEstimator
if not os.path.exists("data"):
    os.makedirs("data")

# MadMiner output
logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.INFO,
)



# Defining values for the simulation
N_wilson = 1
N_obs = 1
benchmark_points = int((N_wilson**2+2+3*N_wilson)/2) # This is just consequence of N_w, N_obs

mus = np.asarray([np.linspace(-2+j/3,2-j/3,N_obs) for j in range(N_wilson+1)])
SIGMAS = np.identity(mus.shape[1])

# This defines some standard x_range. At worst roughly we are talking about [-15,15] per dimension (-10-5*sigmas,10+5*sigmas).
# We restrict the simulation to this box

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
        funs = np.asarray([multivariate_normal.pdf(xs,mean=m,cov=SIGMAS) for m in mus])
        return ((th*funs).sum())**2
    if ipt.ndim == 2:
        th = ipt[:,0:N_wilson]
        th = np.column_stack((np.ones(ipt.shape[0]),th))
        xs = ipt[:,N_wilson:N_wilson+N_obs]
        funs = np.asarray([multivariate_normal.pdf(xs,mean=m,cov=SIGMAS) for m in mus])
        return ((th*funs.T).sum(-1))**2

def dsigma_0(ipt_x_only):
    th = np.zeros(N_wilson)
    th = np.insert(th,0,1)
    xs = ipt_x_only
    funs = np.asarray([multivariate_normal.pdf(xs,mean=m,cov=SIGMAS) for m in mus])
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
        #import ipdb; ipdb.set_trace()
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

#import ipdb; ipdb.set_trace()

#Compute sigmas coefficients; structure is s0+s1th+...; Find polynomial coming from square from matrix multiplication thing
def coefficients_vector(theta):
    th = theta
    th = np.insert(th,0,1)
    mat = th[:,np.newaxis]@th[np.newaxis,:]
    mat[np.triu_indices(N_wilson+1, k = 1)]*=2
    return mat[np.triu_indices(N_wilson+1)]


def fun_to_minim(basis):
    basis = np.reshape(basis,(int((N_wilson**2+2+3*N_wilson)/2),N_wilson))
    mat = []
    for elem in basis:
        mat.append(coefficients_vector(elem))
    mat = np.asarray(mat)
    vec = mat[:,0]
    return np.sum(vec**2)


#Decide sigma basis. I am just looking at the error on the first coefficient which is clearly the heaviest source
sigma_basis = (minimize(fun_to_minim,np.random.uniform(-10,10,int((N_wilson**2+2+3*N_wilson)/2)*N_wilson)))["x"]

sigma_basis = np.reshape(sigma_basis,(int((N_wilson**2+2+3*N_wilson)/2),N_wilson))

matrix_system = []
for elem in sigma_basis:
    matrix_system.append(coefficients_vector(elem))

inverse = np.linalg.inv(matrix_system)

sigma_values = []
for elem in sigma_basis:
    sigma_values.append(simulation(100000,random_theta = False, theta = elem)[1])

coefficients = inverse@sigma_values

np.save("sigma_coefficients.npy",coefficients)

##Also here: minimize at first order for the ratio error propagation

def sigma_at_theta(theta):
    return coefficients@coefficients_vector(theta)

print(f"This is just a check: sigma simulated {sigma_values[0]}")
print(f"Sigma recomputed by function {sigma_at_theta(sigma_basis[0])}" )

#Check a plot

th_test = np.zeros(N_wilson-1)
ys = []
xs = np.linspace(-20,20,100)
for el in xs:
    ys.append(sigma_at_theta(np.insert(np.zeros(N_wilson-1),0,el)))

plt.plot(xs,ys)
plt.savefig("thetacheck")

def fun_to_minim_ratio(basis):
    basis = np.reshape(basis,(int((N_wilson**2+2+3*N_wilson)/2),N_wilson))
    matrix = []
    for j,elem in enumerate(basis):
        matrix.append(coefficients_vector(elem)/sigma_at_theta(elem))
        #print(f"This is sigma at the theta {elem} ",sigma_at_theta(elem))
    matrix = np.asarray(matrix)
    inverse = np.linalg.inv(matrix)
    vec = inverse[0,:]
    return np.sum(vec**2)

print(minimize(fun_to_minim_ratio,np.random.uniform(-5,5,int((N_wilson**2+2+3*N_wilson)/2)*N_wilson))["x"])
ratio_basis = minimize(fun_to_minim_ratio,np.random.uniform(-5,5,int((N_wilson**2+2+3*N_wilson)/2)*N_wilson))["x"].reshape((int((N_wilson**2+2+3*N_wilson)/2),N_wilson))

np.save("bench_basis.npy", ratio_basis)

simulations = []
sigmas = []


size_bench_simulation = 10000


total_bench_size = (benchmark_points+1)*size_bench_simulation

size_random_sim = int(total_bench_size/2)


for elem in ratio_basis:
    sims = simulation(size_bench_simulation,random_theta = False, theta = elem)
    simulations.append(sims[0][:,N_wilson:N_wilson+N_obs])
    sigmas.append(sims[1])

x_from_theta1 =  simulation(size_bench_simulation,False,np.zeros(N_wilson))[0][:,N_wilson:N_wilson+N_obs]
for (j,elem) in enumerate(simulations):
    x_train = np.hstack((elem, x_from_theta1)).reshape(-1, N_obs)
    y_train = np.hstack((np.zeros(elem.shape[0]), np.zeros(x_from_theta1.shape[0]))).reshape(-1, 1)
    np.save(f"data/x_train_{j}.npy", x_train)
    np.save(f"data/y_train_{j}.npy", y_train)








    
