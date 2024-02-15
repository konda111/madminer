import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.optimize import minimize



class Model:
    def __init__(self, N, D, m, s,theta_l,theta_r):
        self.N = N #number of observables
        self.D = D #number of Wilson coefficient dimensions
        self.m = m 
        self.s = s #These are the means and sigmas for multidimensional case. These should have the following shape:
        # (#coefficients, N)
        # coefficients is given by: 
        self.theta_l = theta_l
        self.theta_r = theta_r
        self.r = (np.max(self.m)+3*np.max(self.s))
        self.l = (np.min(self.m)-3*np.max(self.s))
        th = np.random.uniform(self.theta_l,self.theta_r,(1000,self.D))
        z = np.random.uniform(self.l,self.r,(1000,self.N))
        M = 0
        for (ths,zs) in zip(th,z):
            f = self.unnormed_prob(zs,ths)
            if M < f:
                M = f
        self.max = f



    def coefficients(self,theta):
        return np.insert(theta,0,1) #Dimension is D+1
    def weights(self,theta):
        coeffs = self.coefficients(theta)
        matrix = coeffs[:,np.newaxis]@coeffs[np.newaxis,:]
        upper = matrix[np.triu_indices(self.D+1)]
        return upper
    def unnormed_prob(self,z,theta):
        p = []
        for (a,b) in zip (self.m,self.s):
            #import ipdb; ipdb.set_trace()
            p.append(multivariate_normal.pdf(z,mean=a,cov=b))
        return (np.asarray(p)@self.weights(theta))**2
    
    def simulation_random_theta(self,n_reps):
        eff_n = 0
        z_events = []
        thetas = []
        while eff_n < n_reps:
            z = np.random.uniform(self.l,self.r,self.N)
            theta = np.random.uniform(self.theta_l,self.theta_r,self.D)
            p = self.unnormed_prob(z,theta)
            if p < self.max:
                z_events.append(z)
                thetas.append(theta)
                eff_n += 1
        return np.asarray(z_events),np.asarray(thetas)
    
    def simulation_fixed_theta(self,n_reps,th):
        eff_n = 0
        tot = 0
        z_events = []
        thetas = []
        while eff_n < n_reps:
            z = np.random.uniform(self.l,self.r,self.N)
            theta = th
            p = self.unnormed_prob(z,theta)
            if p < self.max:
                z_events.append(z)
                thetas.append(theta)
                eff_n += 1
            tot += 1
        return np.asarray(z_events),np.asarray(thetas), eff_n/tot


# Define parameters for simulation
N = 4 #Number of observables
D = 2 #Number of Wilson coefficients
L = int(((D+1)**2+D+1)/2)

modello = Model(N,D,np.random.uniform(1,2,(L,N)),np.random.uniform(1,2,(L,N)),-2,2)