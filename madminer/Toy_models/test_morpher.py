import os
import logging
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

from madminer.ml import ParameterizedRatioEstimator, MorphParameterizedRatioEstimator
if not os.path.exists("data"):
    os.makedirs("data")

# MadMiner output
logging.basicConfig(
    format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
    datefmt="%H:%M",
    level=logging.INFO,
)


carl = MorphParameterizedRatioEstimator(n_hidden=(20, 20))

carl.train(
    method="carl",
    x="data/x_train.npy",
    y="data/y_train.npy",
    n_epochs=20,
)

carl.save("models/carl")