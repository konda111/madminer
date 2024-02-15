import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Model import Model as m

class Weight_optimization:
    def __init__(self,N, D,theta_l,theta_r):
        self.N = N
        self.D = D
        self.theta_l = theta_l
        self.theta_r = theta_r

    def coefficients(self,theta):
        return np.insert(theta,0,1)
    
    def weights(self,theta):
        coeffs = self.coefficients(theta)
        matrix = coeffs[:,np.newaxis]@coeffs[np.newaxis,:]
        upper = matrix[np.triu_indices(self.D+1)]
        return upper
    
    def sigma_coeffs(self, sigma_basis, sigmas):
        m = []
        for (j,elem) in enumerate(sigma_basis):
            m.append(self.weights(elem))
        inv = np.linalg.inv(m)
        return inv@sigmas

    def whole_expression(self,theta,basis,sigma_coeffs,sigma_basis):
        matrix_num = []
        for (j,elem) in enumerate(basis):
            matrix_num.append(self.weights(elem)/(self.weights(elem)@sigma_coeffs))
        num_inverse = np.linalg.inv(matrix_num)
        numerator = self.weights(theta)@num_inverse
        denominator = self.weights(theta)@sigma_coeffs
        return np.sum(numerator/denominator**2)
    
    def sigma_error(self,th,inverse,values):
        return self.weights(th)@inverse@values
    
    def min_task(self,sigma_basis):
        L = int(((self.D+1)**2+self.D+1)/2)
        sigma_basis = np.reshape(sigma_basis,(L,self.D))
        m = []
        for (j,elem) in enumerate(sigma_basis):
            m.append(self.weights(elem))
        inv = np.linalg.inv(m)
        ths = np.reshape(list(np.linspace(-4,4,10))*self.D,(self.D,10))
        mesh = np.vstack(np.meshgrid(*ths)).reshape(self.D,-1).T
        M = 0
        for elem in mesh:
            s = np.sum(self.sigma_error(elem,inv)**2)
            if s > M:
                M = s
        return M
        
        
        
Ds = [3]  
best_basis = 10
rgs = [2,3,10,100,1000]
for r in rgs:
    for D in Ds:
        # Define parameters for simulation
        N = 4 #Number of observables
        #D is Number of Wilson coefficients
        print(D)
        L = int(((D+1)**2+D+1)/2)
        model = m(N,D,np.random.uniform(1,2,(L,N)),np.random.uniform(1,2,(L,N)),-r,r)
        w = Weight_optimization(N,D,-3,3)
        #print(w.weights(np.random.uniform(-5,5,D)))
        basis_1 = np.random.uniform(-10,10,(L,D))
        basis_2 = np.random.uniform(-10,10,(L,D))

        t = np.reshape(list(np.linspace(-4,4,10))*D,(D,10))
        ths = np.vstack(np.meshgrid(*t)).reshape(D,-1).T
        print("ths shape",np.shape(ths))
        true = []
        for elem in ths:
            true.append(np.sum(w.weights(elem)))

        devs = []
        best_std = 10
        for h in range(1):
            reg_1 = []
            reg_2 = []
            reg_3 = []
            basis_1 = np.random.uniform(-10,10,(L,D))
            #basis_2 = np.random.uniform(-10,10,(L,D))
            for pippo in range(100):
                s_1 = []
                s_2 = []
                for elem in basis_1:
                    s_1.append(np.sum(w.weights(elem)*np.random.normal(1,0.001)))
                #for elem in basis_2:
                    #s_2.append(np.sum(w.weights(elem))*np.random.normal(1,0.001))
                mat_1 = []
                for (j,elem) in enumerate(basis_1):
                    mat_1.append(w.weights(elem))
                inv_1 = np.linalg.inv(mat_1)
                mat_2 = []
                #for (j,elem) in enumerate(basis_2):
                    #mat_2.append(w.weights(elem))
                #inv_2 = np.linalg.inv(mat_2)
                for (j,elem) in enumerate(ths):
                    reg_1.append((w.sigma_error(elem,inv_1,s_1)-true[j])/true[j])
                    #reg_2.append((w.sigma_error(elem,inv_2,s_2)-true[j])/true[j])
                    #reg_3.append((true[j]*np.random.normal(1,0.001)-true[j])/true[j])




            plt.hist(np.asarray(reg_1)/0.001,label="regression_1",alpha=0.3)
            #plt.hist(reg_2,label="regression_2",alpha=0.3)
            #plt.hist(reg_3,label="normal_err",alpha=0.3)
            #plt.xscale("log")
            #plt.yscale("log")
            #plt.legend()
            plt.savefig(f"figs/Test_{h}_{r}")
            plt.clf()
            print(np.shape(reg_1))
            s = np.std(np.asarray(reg_1)/0.001)
            devs.append(np.std(reg_1))
            if s < best_std:
                best_std = s
                best_basis = basis_1
            #print(np.std(reg_2))
            #print(np.std(reg_3))
        print(devs)
        plt.hist(devs,bins=60)
        plt.savefig(f"figs/Dimension={D}_range_th={r}")
        plt.clf()

print(best_basis)