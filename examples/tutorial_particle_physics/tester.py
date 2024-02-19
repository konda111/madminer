import logging
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


from madminer.sampling import SampleAugmenter
from madminer import sampling
from madminer.ml import ParameterizedRatioEstimator, BayesianParameterizedRatioEstimator
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

sampler = SampleAugmenter("data/lhe_data_shuffled.h5")
x, theta0, theta1, y, r_xz, t_xz, n_effective = sampler.sample_train_ratio(
    theta0=sampling.random_morphing_points(100, [("gaussian", 0.0, 0.5), ("gaussian", 0.0, 0.5)]),
    theta1=sampling.benchmark("sm"),
    n_samples=5000,
    folder="./data/samples",
    filename="train_ratio",
    sample_only_from_closest_benchmark=True,
    return_individual_n_effective=True,
)
_ = sampler.sample_test(
    theta=sampling.benchmark("sm"),
    n_samples=1000,
    folder="./data/samples",
    filename="test",
)
_, _, neff = sampler.sample_train_plain(
    theta=sampling.morphing_point([0, 0.5]),
    n_samples=10000,
)

estimator = BayesianParameterizedRatioEstimator(n_hidden=(60, 60), activation="tanh")
estimator.train(
    method="alices",
    theta="data/samples/theta0_train_ratio.npy",
    x="data/samples/x_train_ratio.npy",
    y="data/samples/y_train_ratio.npy",
    r_xz="data/samples/r_xz_train_ratio.npy",
    t_xz="data/samples/t_xz_train_ratio.npy",
    alpha=10,
    n_epochs=10,
    scale_parameters=True,
)

estimator.save("models/alices")