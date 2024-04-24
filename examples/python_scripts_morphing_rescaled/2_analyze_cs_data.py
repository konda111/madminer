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
        h = int(os.environ["ARR_NUM"])
        lhe.add_sample(
                    lhe_filename=f"mg_processes/signal_{j}_{h}/Events/run_01/unweighted_events.lhe.gz",
                    sampled_from_benchmark=name,
                    is_background=False,
                    k_factor=1.0,
                )
        lhe.add_observable(
            "met",
            "met.pt",
            required=True,
        )
        lhe.add_observable(
                "pt_j1",
                "j[0].pt",
                required=False,
                default=0.0,
        )
        lhe.add_observable(
                "delta_phi_jj",
                "j[0].deltaphi(j[1]) * (-1.0 + 2.0 * float(j[0].eta > j[1].eta))",
                required=True,
        )
        lhe.add_cut("(a[0] + a[1]).m > 122.0")
        lhe.add_cut("(a[0] + a[1]).m < 128.0")
        lhe.add_cut("pt_j1 > 20.0")
        lhe.analyse_samples()
        lhe.save(f"data/generation_{h}/lhe_data_{j}.h5")
        sampler = SampleAugmenter(f"data/generation_{h}/lhe_data_{j}.h5")
        np.save(f"data/generation_{h}_cross_sections/cross_section_{j}.npy",sampler.cross_sections(sampling.benchmark(name))[1][0])