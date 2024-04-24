import logging
import sys
import os
sys.path.append('/remote/gpu02/crescenzo/MadMiner')
from madminer.lhe import LHEReader
from madminer.sampling import SampleAugmenter
from madminer import sampling

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
miner = MadMiner()

miner.load("data/setup.h5")

cross_sections = []
for j,name in enumerate(list(miner.benchmarks.keys())):
    lhe = LHEReader("data/setup.h5")
    if j == int(os.environ["NUM"]):
        lhe.add_sample(
                    lhe_filename=f"mg_processes/signal_{j}/Events/run_01/unweighted_events.lhe.gz",
                    sampled_from_benchmark=name,
                    is_background=False,
                    k_factor=1.0,
                )
        lhe.add_observable(
                "met",
                "met.pt",
                required=True,
            )
        lhe.analyse_samples()
        lhe.save(f"data/lhe_data_{j}.h5")
        sampler = SampleAugmenter(f"data/lhe_data_{j}.h5")
        np.save(f"data/cross_section_{j}.npy",sampler.cross_sections(sampling.benchmark(name))[1][0])