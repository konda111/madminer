import logging
import sys
sys.path.append('/remote/gpu02/crescenzo/MadMiner')
from madminer.sampling import combine_and_shuffle
from madminer.core import MadMiner
from madminer.ml import MorphParameterizedRatioEstimator
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
for j in range(1,6):
    combine_and_shuffle([f"data/lhe_training_data_{j}.h5",f"data/lhe_training_data_0.h5"], f"data/lhe_data_training_{j}_shuffled.h5")
for j in range(1,6):
    sampler = SampleAugmenter(f"data/lhe_data_training_{j}_shuffled.h5")
    x, theta0, theta1, y, r_xz, t_xz, n_effective = sampler.sample_train_ratio(
                theta0=sampling.benchmark(f"morphing_basis_vector_{j}"),
                theta1=sampling.benchmark("sm"),
                n_samples=200,
                folder=f"./data/samples",
                filename=f"train_ratio_bench_{j}",
                sample_only_from_closest_benchmark=True,
                return_individual_n_effective=True,
    )