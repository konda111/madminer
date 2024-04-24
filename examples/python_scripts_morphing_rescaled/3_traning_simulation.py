import logging
import sys 
sys.path.append('/remote/gpu02/crescenzo/MadMiner')
from madminer.lhe import LHEReader

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

from madminer import MadMiner
import numpy as np

cross_sections = []

for j in range(6):
    cross_sections.append(float(np.load(f"data/generation_5_cross_sections/cross_section_{j}.npy")))

cross_sections = np.asarray(cross_sections)
reduced_cs = cross_sections[1:cross_sections.size]/cross_sections[0]
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

miner.load("data/setup.h5")
miner.morpher.set_reduced(reduced_cs)

miner.ratio_set_morphing(max_overall_power=2, lim = 200)
#import ipdb; ipdb.set_trace()
mg_dir = "/remote/gpu02/crescenzo/MG5_aMC_v3_5_3"
for j,elem in enumerate(miner.benchmarks.keys()):
    miner.run(
    sample_benchmark=elem,
    mg_directory=mg_dir,
    mg_process_directory=f"./mg_processes/ratio_signal_{j}",
    proc_card_file="cards/proc_card_signal.dat",
    param_card_template_file="cards/param_card_template.dat",
    run_card_file="cards/run_card_signal_ratio.dat",
    log_directory=f"logs/signal_{j}",
    python_executable="python3",
    only_prepare_script=True
)

miner.save("data/ratio_setup.h5")