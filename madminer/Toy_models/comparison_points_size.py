import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
Total = 300000

throws = [3,100,200,300]

def true_fun(x):
    return x**2+x+1

def function_to_min(x):
    m = [[x[0]**2,x[0],1],
         [x[1]**2,x[1],1],
         [x[2]**2,x[2],1]]
    inv = np.linalg.inv(m)**2
    vec = np.asarray([x[0]**2+x[0]+1,
                      x[1]**2+x[1]+1,
                      x[2]**2+x[2]+1])**2
    return np.sum(inv@vec)
    
for j in range(10):
    minimal = minimize(function_to_min,np.random.uniform(-300,300,3))
    print(minimal["x"])
    print(minimal["fun"])
    basis = minimal["x"]
    ast = []
    bst = []
    cst = []
    for h in range(10000):
        ys = true_fun(basis)
        shift = np.random.multivariate_normal(np.full_like(ys,0),np.diag(np.abs(ys)/np.sqrt(Total/3)))
        measured_ys = ys+shift
        a,b,c = np.polyfit(basis,measured_ys,2)
        ast.append(a)
        bst.append(b)
        cst.append(c)
    plt.hist(ast,label=f"Mean:{np.mean(ast)}, standard dev:{np.std(ast)}",bins=200)
    plt.legend()
    plt.savefig(f"Python minimizer trial {j}:a")
    plt.clf()
    plt.hist(bst,label=f"Mean:{np.mean(bst)}, standard dev:{np.std(bst)}",bins=200)
    plt.legend()
    plt.savefig(f"Python minimizer trial {j}:b")
    plt.clf()
    plt.hist(cst,label=f"Mean:{np.mean(cst)}, standard dev:{np.std(cst)}",bins=200)
    plt.legend()
    plt.savefig(f"Python minimizer trial {j}:c")
    plt.clf()



print(function_to_min(np.asarray([-0.9264,0.2834,-1*10**7])))
ast = []
bst = []
cst = []
basis = np.asarray([-0.9264,0.2834,-1*10**7])
for h in range(10000):
    ys = true_fun(basis)
    shift = np.random.multivariate_normal(np.full_like(ys,0),np.diag(np.abs(ys)/np.sqrt(Total/3)))
    measured_ys = ys+shift
    a,b,c = np.polyfit(basis,measured_ys,2)
    ast.append(a)
    bst.append(b)
    cst.append(c)
plt.hist(ast,label=f"Mean:{np.mean(ast)}, standard dev:{np.std(ast)}",bins=200)
plt.legend()
plt.savefig(f"Optimal_A?")
plt.clf()
plt.hist(bst,label=f"Mean:{np.mean(bst)}, standard dev:{np.std(bst)}",bins=200)
plt.legend()
plt.savefig(f"Optimal_B?")
plt.clf()
plt.hist(cst,label=f"Mean:{np.mean(cst)}, standard dev:{np.std(cst)}",bins=200)
plt.legend()
plt.savefig(f"Optimal_C?")
plt.clf()


"""for t in throws:
    ast = []
    bst = []
    cst = []
    xs = np.random.uniform(-50,50,t)
    if t == 3:
        xs = np.asarray([-0.924,0.2834,-1*10**7])
    for j in range(1000):
        ys = true_fun(xs)
        shift = np.random.multivariate_normal(np.full_like(ys,0),np.diag(np.abs(ys)/np.sqrt(Total/t)))
        measured_ys = ys+shift
        a,b,c = np.polyfit(xs,measured_ys,2)
        ast.append(a)
        bst.append(b)
        cst.append(c)
    plt.hist(ast,label=f"Mean:{np.mean(ast)}, standard dev:{np.std(ast)}")
    plt.legend()
    plt.savefig(f"Number of points:{t}")

    plt.clf()
    print(t)"""

