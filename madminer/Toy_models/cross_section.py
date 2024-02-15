import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
A = 1
B = 1
C = 1

"""def fisher_mat(th):
    m = [[1/(1+th[0]**2)+1/(1+th[1]**2),th[0]**2/(1+th[0]**2)+th[1]**2/(1+th[1]**2)],
         [th[0]**2/(1+th[0]**2)+th[1]**2/(1+th[1]**2),th[0]**4/(1+th[0]**2)+th[1]**4/(1+th[1]**2)]]
    
    return np.trace(np.linalg.inv(m))


minim = minimize(fisher_mat,np.random.uniform(-5,5,2),method="Nelder-Mead")
print(minim)"""
def true(x,A,B,C):
    return A + B*x + C*x*x
maxes = []
for t in range(10):
    sizes = [3,15,27,60,120] 
    bases = []
    total_points = 120*9*1000
    for s in sizes:
        if s == 3:
            bases.append(np.random.uniform(-30,30,s))
        elif s == 15:
            bases.append(np.random.uniform(-30,30,s))
        else:
            bases.append(np.random.uniform(-30,30,s)) #define base for each thing
        
            


    fits = []
    for (j,base) in enumerate(bases):
        y_vals = []
        for (h,elem) in enumerate(base):
            y_vals.append(np.mean(np.random.normal(true(elem,A,B,C),10,int(total_points/sizes[j]))))
        fits.append(np.polyfit(base,y_vals,2))

    x = np.linspace(-20,20,100)
    #plt.plot(x,true(x,A,B,C),label="True law")
    for (j,f) in enumerate(fits):
        plt.plot(x,np.abs((true(x,f[2],f[1],f[0])-true(x,A,B,C))/true(x,A,B,C)),label=f"base={np.shape(bases[j])}")
        maxes.append(np.max(np.abs((true(x,f[2],f[1],f[0])-true(x,A,B,C))/true(x,A,B,C))))
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"trial number {t}")
    plt.clf()


