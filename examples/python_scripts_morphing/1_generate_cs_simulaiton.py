import logging
import sys
import matplotlib.pyplot as plt
import numpy as np


sys.path.append('/remote/gpu02/crescenzo/MadMiner')
from madminer import MadMiner
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

miner = MadMiner()

miner.add_parameter(
    lha_block="dim6",
    lha_id=2,
    parameter_name="CWL2",
    morphing_max_power=2,
)
miner.add_parameter(
    lha_block="dim6",
    lha_id=5,
    parameter_name="CPWL2",
    morphing_max_power=2,
)

miner.cs_set_morphing(max_overall_power=2, lim = 20)
miner.save("data/setup.h5")

mg_dir = "/remote/gpu02/crescenzo/MG5_aMC_v3_5_3"

"""for j,elem in enumerate(miner.benchmarks.keys()):
        miner.run(
        sample_benchmark=elem,
        mg_directory=mg_dir,
        mg_process_directory=f"./mg_processes/signal_{j}",
        proc_card_file="cards/proc_card_signal.dat",
        param_card_template_file="cards/param_card_template.dat",
        run_card_file=f"cards/run_card_signal.dat",
        log_directory=f"logs/signal_{j}",
        python_executable="python3",
        only_prepare_script=True
)"""


ws = []
cs_basis = miner.morpher.cs_basis
#import ipdb; ipdb.set_trace()
th0,th1 = np.meshgrid(np.linspace(-20,20,50),np.linspace(-20,20,50))
grid = np.vstack((th0.flatten(),th1.flatten())).T
grid = np.asarray(grid)
for elem in grid:
    ws.append(np.sqrt(np.sum(miner.morpher.compute_weight_no_sigma_2(elem,cs_basis[1:6])**2)))
plt.scatter(grid[:,0],grid[:,1],c=ws )
plt.colorbar(label="CS morphing weights norm")
plt.scatter(cs_basis[:,0],cs_basis[:,1],s=100,marker="*",c="r",label="Basis points")
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$")
plt.legend()
plt.savefig("plots/cs_morphing")
plt.clf()



radii = np.linspace(50,1000,5)
for r in radii:
    cs_basis = [[0,0]]
    for j in range(5):
        cs_basis.append([r*np.cos(2*j/5*np.pi),r*np.sin(2*j/5*np.pi)])
    cs_basis = np.asarray(cs_basis)
    ws = []
    for elem in grid:
        ws.append(np.sqrt(np.sum(miner.morpher.compute_weight_no_sigma_2(elem,cs_basis[1:6])**2)))
    plt.scatter(grid[:,0],grid[:,1],c=ws )
    plt.colorbar(label="CS morphing weights norm")
    plt.scatter(cs_basis[:,0],cs_basis[:,1],s=100,marker="*",c="r",label="Basis points")
    plt.xlim([-200,200])
    plt.ylim([-200,200])
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    plt.legend()
    plt.savefig(f"plots/cs_morphing_pentagon_{r}.png")
    plt.clf()
