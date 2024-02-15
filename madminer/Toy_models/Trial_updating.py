import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.optimize import minimize


trials = [1000,2000,3000,5000,10000,50000,10000]

t_A = 2
t_B = -4
t_C = 5

def true_function(x):
    return t_A*x*x+t_B*x+t_C


basis = np.random.uniform(-10,10,3)
ys = true_function(basis)
shift = np.random.multivariate_normal(ys,np.diag(np.abs(ys)/np.sqrt(1000)))
print("shift",shift)
measured_ys = ys+shift



a,b,c = np.polyfit(basis,measured_ys,2)

ast = []
bst = []
cst = []
"""for j in range(100):
    s0 = np.random.normal(ys[0],np.abs(ys[0]))
    s1 = np.random.normal(ys[1],np.abs(ys[1]))
    s2 = np.random.normal(ys[2],np.abs(ys[2]))
    measured_ys = [ys[0]+s0,ys[1]+s1,ys[2]+s2]
    a,b,c = np.polyfit(basis,measured_ys,2)
    ast.append(a)
    bst.append(b)
    cst.append(c)

print("first a mean with random basis",np.mean(ast))
print("first b mean with random basis",np.mean(bst))
print("first c mean with random basis",np.mean(cst))"""



a_test, b_test, c_test = np.polyfit(basis,ys,2)
print("should be perfect:", a_test, b_test, c_test)

general_bases = []
general_bases.append(basis)

for n in trials:
    #assume the correct a,b,c are the ones computed in the step before. 
    #for now disregard any info coming from the previous ones (-2,-3 etc)
    def error(x):
        m = [[x[0]**2,x[0],1],
             [x[1]**2,x[1],1],
             [x[2]**2,x[2],1]]
        ## Find condition number. If it is not decent then discard
        evals = np.linalg.eigvals(m)
        ratio = np.abs(np.max(evals)/np.min(evals))
        #print("this is ratio",ratio)
        if ratio > 100:
            return float('inf')
        if np.any(np.abs(x)) > 10000:
            return float('inf')
        #print(np.linalg.det(m))
        ys_err = [a*x[0]**2+b*x[0]+c,a*x[1]**2+b*x[1]+c,a*x[2]**2+b*x[2]+c]
        if np.any(np.abs(ys_err)) > 10000:
            return float('inf')
        #print(a,b,c)
        try: 
            inv = np.asarray(np.linalg.inv(m))
        except:
            return float('inf')
        inv = inv**2
        ys_err = np.asarray(ys_err)**2
        variances = inv@measured_ys
        return np.sqrt(np.sum(variances))
    #find next basis. Start from the previous basis, it should make sense in the long run
    minimal = minimize(error,basis,method="Nelder-Mead")
    value_err = minimal["fun"]
    print("this is error value", value_err)
    basis = minimal["x"]

    #Compute updated coefficients with less error now given by n in trials
    ys = true_function(basis)
    print("SQRT", np.sqrt(n))
    print("Ys", ys)
    shift = np.random.multivariate_normal(ys,np.diag(np.abs(ys)/np.sqrt(n)))
    print("This is shifts",shift)

    measured_ys = ys+shift
    a,b,c = np.polyfit(basis,measured_ys,2)/2

    print("abcs first run", a,b,c)
    general_bases.append(basis)



deviations_a = []
deviations_b = []
deviations_c = []
means_a = []
means_b = []
means_c = []
for bx in general_bases:
    list_as = []
    list_bs = []
    list_cs = []
    #print("basis: ", bx)
    for j in range(1000):
        ys = true_function(bx)
        shift = np.random.multivariate_normal(ys,np.diag(np.abs(ys)))
        measured_ys = ys+shift
        a0,b0,c0 = np.polyfit(basis,measured_ys,2)/2
        list_as.append(a0)
        list_bs.append(b0)
        list_cs.append(c0)
    deviations_a.append(np.std(list_as))
    deviations_b.append(np.std(list_bs))
    deviations_c.append(np.std(list_cs))
    means_a.append(np.mean(list_as))
    means_b.append(np.mean(list_bs))
    means_c.append(np.mean(list_cs))


plt.plot(deviations_a)
plt.yscale("log")
plt.savefig("A")
plt.clf()
plt.plot(deviations_b)
plt.yscale("log")
plt.savefig("B")
plt.clf()
plt.plot(deviations_c)
plt.yscale("log")
plt.savefig("C")
plt.clf()
plt.plot(means_a)
#plt.yscale("log")
plt.savefig("A_means")
plt.clf()
plt.plot(means_b)
#plt.yscale("log")
plt.savefig("B_means")
plt.clf()
plt.plot(means_c)
#plt.yscale("log")
plt.savefig("C_means")
plt.clf()

print(means_a[-1],means_b[-1],means_c[-1])
    







